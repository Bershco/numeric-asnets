"""Code for training on several problems at the same time. Their all live in
their own sandboxed Python interpreters so that they can have their own copies
of MDPSim and SSiPP."""

import ctypes
import logging
import multiprocessing
import signal
import sys
from pathlib import Path
from time import sleep, time
import weakref

import os, json, socket, rpyc, atexit
import uuid, getpass

import joblib
import numpy as np
from rpyc import OneShotServer, BaseNetref
import types
import collections
import dataclasses
from rpyc.utils.server import ThreadedServer
import tensorflow as tf
from typing import Callable

import asnets.models
from asnets.models import PropNetworkWeights, make_network
from asnets.supervised import WeightedReplayBuffer
from asnets.utils.rpyc_utils import to_local, _shutdown_proc

rpyc.core.protocol.DEFAULT_CONFIG['allow_getattr'] = True
rpyc.core.protocol.DEFAULT_CONFIG['allow_setattr'] = True
rpyc.core.protocol.DEFAULT_CONFIG['allow_delattr'] = True
rpyc.core.protocol.DEFAULT_CONFIG['safe_attrs'].add("copy")
rpyc.core.protocol.DEFAULT_CONFIG['safe_attrs'].add("sizeof")
rpyc.core.protocol.DEFAULT_CONFIG['safe_attrs'].add("__sizeof__")
from asnets.utils.prof_utils import try_save_profile
from asnets.utils.py_utils import set_random_seeds

from rpyc.utils.classic import obtain



def parent_death_pact(signal: signal.Signals=signal.SIGINT) -> None:
    """Commit to kill current process when parent process dies. This function
    only works on linux for now. Specifically, it calls prctl() with the
    operation PR_SET_PDEATHSIG, which is documented in the kernel source code
    in include/uapi/linux/prctl.h. This operation is available for
    Linux>=2.1.57. 

    Args:
        signal: the signal to send to the current process when the parent
        process dies. Defaults to SIGINT.
    """
    assert sys.platform == 'linux', \
        "this fn only works on Linux right now"
    libc = ctypes.CDLL("libc.so.6")
    # see include/uapi/linux/prctl.h in kernel
    PR_SET_PDEATHSIG = 1
    # last three args are unused for PR_SET_PDEATHSIG
    retcode = libc.prctl(PR_SET_PDEATHSIG, signal, 0, 0, 0)
    if retcode != 0:
        raise Exception("prctl() returned nonzero retcode %d" % retcode)

def start_server(service_args: 'ProblemServiceConfig',unix_socket_path: str = None) -> None:
    """Start a OneShotServer with the given service_args and socket_path. This
    function is intended to be run in a subprocess. It will set the random seed,
    set up the problem service, and start the OneShotServer. It will also
    save kernprof profile data if it can.

    Args:
        service_args (ProblemServiceConfig): config for the problem service.
        # socket_path (str): path to the socket to host the problem
        service on.
    """

    if service_args.random_seed is not None:
        set_random_seeds(service_args.random_seed)
    # avoid import cycle
    from asnets.supervised import make_problem_service
    parent_death_pact(signal=signal.SIGKILL)
    new_service = make_problem_service(service_args, set_proc_title=True)

    protocol_config = {
        "allow_all_attrs": True,  # Default: False, restricts attributes
        "allowed_attrs": {"copy"},  # Add 'copy' to the list of allowed attributes
        "allow_setattr": True,  # Allow setting attributes if needed
        "allow_delattr": True,  # Allow deleting attributes if needed
    }
    rpyc.core.protocol.DEFAULT_CONFIG.update({
        # this is required for rpyc to allow pickling
        'allow_pickle': True,
        # required for some large problems where get_action() (passed as
        # synchronous callback to child processes) can take a very long time
        # the first time it is called
        'sync_request_timeout': 1800,
    })

    # print(f"READY RPyC {host}:{port}", flush=True)
    print(f"READY RPyC {unix_socket_path}", flush=True)

    import socket

    _old_close = socket.socket.close

    def _debug_close(self, *a, **kw):
        print(f"[DEBUG SOCKET CLOSE] {self} pid={os.getpid()}", flush=True)
        return _old_close(self, *a, **kw)

    socket.socket.close = _debug_close

    _old_serve_client = rpyc.utils.server.ThreadedServer._serve_client

    def _debug_serve_client(self, sock, credentials):
        import traceback, os
        try:
            print(f"[DEBUG WORKER START] {sock} pid={os.getpid()}", flush=True)
            return _old_serve_client(self, sock, credentials)
        except Exception as e:
            print(f"[DEBUG WORKER CRASH] {e}", flush=True)
            traceback.print_exc()
            raise
        finally:
            print(f"[DEBUG WORKER END] {sock} pid={os.getpid()}", flush=True)

    rpyc.utils.server.ThreadedServer._serve_client = _debug_serve_client

    server = ThreadedServer(
        new_service,
        # hostname=host,
        # port=port,
        socket_path=unix_socket_path,
        reuse_addr=True,
        backlog=1024,
        protocol_config=protocol_config,
    )

    print(f'Child process starting {server.__class__.__name__} {server}')

    import psutil, threading, gc

    def monitor():
        p = psutil.Process(os.getpid())
        while True:
            sleep(10)
            mem = p.memory_info().rss / 1e6
            fds = p.num_fds() if hasattr(p, "num_fds") else None
            threads = len(threading.enumerate())
            async_objs = [o for o in gc.get_objects() if isinstance(o, rpyc.core.async_.AsyncResult)]
            print(f"[DEBUG MONITOR] mem={mem:.1f}MB fds={fds} threads={threads}, active async results={len(async_objs)}", flush=True)

    # threading.Thread(target=monitor, daemon=True).start()


    try:
        print(f"[DEBUG] Server entering main loop (PID={os.getpid()})", flush=True)
        server.start()
        print(f"[DEBUG] Server main loop exited normally (PID={os.getpid()})", flush=True)
    except Exception as e:
        import traceback, sys
        print(f"[SERVER ERROR] Exception in server.start(): {e}", flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        print("[SERVER] Shutting down cleanly", flush=True)

def wait_exists_polling(file_path: str, 
                        max_wait: float, delta: float=0.05) -> bool:
    """Check if file exists every delta seconds.
    
    Args:
        file_path (str): path to file to check.
        max_wait (float): maximum time to wait for file to exist.
        delta (float): how long to wait between checks. Defaults to 0.05.
    
    Returns:
        bool: true if file exists, False if we timed out.
    """
    start_time = time()
    while not os.path.exists(file_path):
        sleep(delta)
        if time() - start_time > max_wait:
            return False
    return True

# --- small helpers ---
def _wait_for_addr(path: str, timeout: float = 60.0, poll: float = 0.1, proc: "Process|None" = None):
    deadline = time() + timeout
    last_err = None
    while time() < deadline:
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError as e:
            last_err = e
            if proc is not None and not proc.is_alive():
                raise RuntimeError(f"Server exited before writing addr file (exitcode={proc.exitcode}).")
            sleep(poll)
    # optional: nicer message if we saw a crash
    if proc is not None and not proc.is_alive():
        raise RuntimeError(f"Timed out and server is dead (exitcode={proc.exitcode}).")
    raise TimeoutError(f"Timed out waiting for RPYC addr file: {path}") from last_err


def _connect_via_info(info):
    from rpyc.utils.factory import unix_connect
    c = unix_connect(info)
    c.ping()
    return c


class ProblemServer(object):
    """Spools up another process to host a ProblemService."""
    # how long we need to wait for the connection to spool up
    MAX_WAIT_TIME = 120.0

    def __init__(self, service_conf: 'ProblemServiceConfig') -> None:
        """Create a new ProblemServer. This will start a new process that will
        create a ProblemService and host it on a socket. This object will
        connect to that socket and provide a proxy to the ProblemService. The
        connection will be closed when this object is destroyed.

        Socket paths are created in /tmp/asnet-sockets-<username> to avoid
        Linux's 108-char limit on socket paths. The limit does not apply to
        filenames. The username is included to avoid the case where somebody
        else creates the dir and prevents us from writing to it.

        Args:
            service_conf (ProblemServiceConfig): configuration for the
            ProblemService to be hosted.
        """
        self.get_socket()
        addr_file = os.environ.get("RPYC_ADDR_FILE")
        # For SLURM batch usage - usage of unix sockets wrecked the runtime of a lot of experiments,
        # changing to TCP instead. If this doesn't already exist, we'll synthesize another.
        if not addr_file:
            # fallback: keep it under the project dir, but your Slurm sets this already
            addr_file = os.path.join(os.getcwd(), "rpyc_addr",
                                     f".rpyc_addr_{os.getpid()}.json")
            os.makedirs(os.path.dirname(addr_file), exist_ok=True)
            os.environ["RPYC_ADDR_FILE"] = addr_file
        else:
            os.makedirs(os.path.dirname(addr_file), exist_ok=True)
        print(f"[multiprocessing debug] Process about to start with {multiprocessing.get_start_method()} as the start method.")
        self._serve_proc = multiprocessing.Process(
            name='worker',
            target=start_server, args=(
                service_conf,
                self._unix_sock_path,
            ))
        print(f'[DEBUG] Trainer PID: {os.getpid()}')
        self.problem_server_slot_id = service_conf.slot_id


        self._service_conf = service_conf
        self._serve_proc.daemon = False  # ensure child stays attached to parent console
        sys.stdout.flush()
        sys.stderr.flush()  # flush parent before fork
        self._serve_proc.start()
        self._addr_file = addr_file

        # info = _wait_for_addr(self._addr_file, proc=self._serve_proc, timeout=self.MAX_WAIT_TIME)
        self._conn = None
        self._bg_thread = None
        self._start_time = time()
        # This makes self._conn the rpyc connection we need.

        # this ensures that we always close connection (& thus terminate server
        # on other end) before shutting down, no matter what
        # (basically weakref.finalize(obj, func) ensures that func is called
        # when obj is destroyed---presumably just beforehand)
        # self._finalizer = weakref.finalize(self._serve_proc, self._kill_conn)
        # self._finalizer = weakref.finalize(self._serve_proc, self.stop)
        self.rb = WeightedReplayBuffer()
        self.curr_prob_name = None
        self.is_first_instance = True

    def connect(self):
        """
        Wait for socket + connect + start BgServingThread.
        Safe to call in parallel from threads.
        """
        if self._conn is not None:
            return

        # Wait for socket to appear (this is the part you want parallelized!)
        # wait_exists_polling(self._unix_sock_path, max_wait=self.MAX_WAIT_TIME)

        # Now do your existing connection logic
        self._get_rpyc_conn()

        # Critical for async: keep client side serving

    # def _kill_conn(self) -> None:
    #     """Close the connection to the server."""
    #     print("[DEBUG] Closing connection to the server through '_kill_conn'.")
    #     if self._conn is not None:
    #         self._conn.close()
    #         self._conn = None

    def stop(self, extra_msg = None) -> None:
        """Close the RPyC connection and stop the server process. Idempotent."""
        # close client connection
        print(f"[DEBUG] {extra_msg if extra_msg is not None else 'Closing connection to the server through stop().'}")
        print(f"[PARENT] Server process (PID={self._serve_proc.pid if self._serve_proc is not None else 'None'}) exited with code {self._serve_proc.exitcode if self._serve_proc is not None else 0}")
        print(f"[DEBUG] Process alive? {self._serve_proc.is_alive() if self._serve_proc is not None else False}")
        try:
            if getattr(self, "_bg_thread", None):
                try:
                    self._bg_thread.stop()
                except Exception:
                    pass
                finally:
                    self._bg_thread = None
            if getattr(self, "_conn", None):
                try:
                    self._conn.close()
                except Exception as e:
                    print(f"Something happened during the closing of the connection.\n Error:{e}")
                finally:
                    self._conn = None
        finally:
            # terminate the server process
            proc = getattr(self, "_serve_proc", None)
            print(f'[DEBUG] Got proc={proc}')
            if proc is not None:
                try:
                    if proc.is_alive():
                        proc.terminate()
                        try:
                            proc.join(timeout=5)
                        except Exception:
                            pass
                        if proc.is_alive():
                            try:
                                os.kill(proc.pid, signal.SIGKILL)
                            except Exception:
                                pass
                            try:
                                proc.join(timeout=5)
                            except Exception:
                                pass
                finally:
                    pass
                    self._serve_proc = None

            # remove the addr file (server also tries via atexit; this is belt & suspenders)
            addr = getattr(self, "_addr_file", None)
            if addr:
                try:
                    os.remove(addr)
                except FileNotFoundError:
                    pass

    def __del__(self):
        """Destructor. This will kill the server process if it's still running.
        """
        if hasattr(self, '_serve_proc') and self._serve_proc is not None:
            print('Cleaning up server process in destructor')
            self.stop()

    def _get_rpyc_conn(self):
        """Get a connection to the server.

        This will create a new connection if one doesn't already exist. In this
        case, it will wait for the server to start with a timeout defined by
        self.MAX_WAIT_TIME. It will also sleep for an additional secton to make
        sure the connection is up.

        Returns:
            The rpyc connection to the server.
        """
        if self._conn is None:
            res = self.MAX_WAIT_TIME - (time() - self._start_time)
            to_wait = max(0.0, res)
            if to_wait > 0:
                # It actually takes a few seconds for the background worker to
                # spool up and start accepting connections. Obviously it could
                # be more than self.MAX_WAIT_TIME, but I don't really have a
                # better way of doing things than this (mostly because all the
                # socket binding in RPyC happens in a monolithic "run
                # everything" method which I can't break up).
                print('Waiting at most %.2fs for rpyc connection' % to_wait)
                # ignore return value; we'll get an error later if the file
                # doesn't exist
                has_sock = wait_exists_polling(
                    self._unix_sock_path, max_wait=to_wait)
                print(f"Wait time up, got has_sock={has_sock}")

            sleep_time = 1.0
            print(f"Sleeping an extra {sleep_time}s to make sure conn is up")
            sleep(sleep_time)
            protocol_config = {
                "allow_all_attrs": True,  # Default: False, restricts attributes
                "allowed_attrs": {"copy"},  # Add 'copy' to the list of allowed attributes
                "allow_setattr": True,  # Allow setting attributes if needed
                "allow_delattr": True,  # Allow deleting attributes if needed
                "sync_request_timeout": None,  # disable watchdog
                "async_request_timeout": None,
                "ping_interval": None,
            }

            self._conn = rpyc.utils.factory.unix_connect(
                path=self._unix_sock_path, config=protocol_config)
            # we can unlink socket after connecting
            print(f"[DEBUG] Connected socket path exists? {os.path.exists(self._unix_sock_path)} before unlink", flush=True)
            os.unlink(self._unix_sock_path)
            self._bg_thread = rpyc.utils.helpers.BgServingThread(self._conn)

        return self._conn

    def set_policy_only(self, value: bool) -> None:
        self.policy_only = value

    def set_enhsp_config(self, config: str) -> None:
        self.enhsp_config = config


    def get_problem_data(self):
        res = self.service.get_problem_data()
        res_local = (to_local(r) for r in res)
        self.obs_dim, self.act_dim, self.dom_meta, self.prob_meta, self.dg_extra_dim = res_local
        return self.obs_dim, self.act_dim, self.dom_meta, self.prob_meta, self.dg_extra_dim

    def get_domain_data(self):
        self.obs_dim, self.act_dim, self.dom_meta, self.prob_meta, self.dg_extra_dim = self.service.get_problem_data()
        return self.dom_meta, self.dg_extra_dim

    def _reconnect(self):
        self._start_new_server()
        # wait_exists_polling(self._unix_sock_path, max_wait=self.MAX_WAIT_TIME)
        self._start_time = time()
        self._get_rpyc_conn()
        return self._conn.root

    def get_socket(self):
        sock_dir = os.path.join(os.getcwd(), "asnet-sockets")
        os.makedirs(sock_dir, exist_ok=True)
        self._unix_sock_path = os.path.join(sock_dir,
                                            'socket.' + uuid.uuid4().hex)

    @property
    def conn(self):
        """Get a connection to the ProblemService. This will create a new
        connection if one doesn't already exist.

        Returns:
            The rpyc connection to the ProblemService.
        """
        return self._get_rpyc_conn()

    @property
    def problem_service(self):
        return self.service

    @property
    def name(self):
        # return self.service.get_current_problem_name()
        return self.curr_prob_name

    @property
    def service(self):
        if self._conn is None:
            raise RuntimeError(
                "Service requested but no active connection. "
                "Did you forget to start/reconnect the worker?"
            )
        return self._conn.root

    def is_conn_alive(self):
        """Return True if RPyC connection still open."""
        try:
            if self._conn is None:
                return False
            return not self._conn.closed
        except Exception as e:
            print(f"[DEBUG] Conn health check failed: {e}")
            return False

    def make_network(self, weight_manager, inner: Callable, args):
        obs_dim, act_dim, dom_meta, prob_meta, dg_extra_dim = self.get_problem_data()
        self.network, weight_manager = inner(
            args,
            obs_dim,
            act_dim,
            dom_meta,
            prob_meta,
            dg_extra_dim,
            weight_manager=weight_manager,
        )
        return weight_manager

    def register_network(self, weights_manager, args):
        weights_manager = to_local(weights_manager)
        self.network_dropout = to_local(args.dropout)
        self.network_debug = to_local(args.net_debug)
        self.network_policy_only = to_local(args.policy_network_only)
        self.get_problem_data()
        self.network, self.weights_manager = make_network(args, self.dom_meta, self.prob_meta, self.dg_extra_dim, weights_manager)
        return self.weights_manager

    def set_weight_manager(self, weight_manager, args):
        hs = args.hidden_size
        num_layers = args.num_layers
        dropout = args.dropout
        print('hidden_size: %d, num_layers: %d, dropout: %f' % (hs, num_layers,
                                                                dropout))
        if weight_manager is not None:
            print('Re-using same weight manager')
        elif args.resume_from:
            print('Reloading weight manager (resuming training)')
            resume_from_str = args.resume_from
            print(f'\n\n[model-loading] - Resuming from: {args.resume_from}\n\n')
            resume_from_str = resume_from_str.replace("\\", '/')  # for Windows support, do not delete.
            resume_from_path_obj = Path(resume_from_str)
            resume_from_path_obj = resume_from_path_obj.resolve(strict=False)
            weight_manager = joblib.load(resume_from_path_obj)
        else:
            print('Creating new weight manager (not resuming)')
            # TODO: should save all network metadata with the network weights or
            # within a separate config class, INCLUDING heuristic configuration
            dom_meta, dg_extra_dim = self.get_domain_data()
            weight_manager = PropNetworkWeights(
                dom_meta,
                hidden_sizes=[(hs, hs)] * num_layers,
                # extra inputs to each action module from data generators
                extra_dim=dg_extra_dim,
                skip=args.skip,
                use_fluents=args.use_fluents,
                use_comparisons=args.use_comparisons)
        return weight_manager

    def _start_new_server(self):
        logging.info(f"[DEBUG] _serve_proc is {self._serve_proc}")
        self.get_socket()
        self._serve_proc = multiprocessing.Process(
            name='worker',
            target=start_server,
            args=(
                self._service_conf,
                self._unix_sock_path,
            )
        )
        self._serve_proc.daemon = False
        self._serve_proc.start()
        logging.info(f"[DEBUG] _serve_proc is {self._serve_proc}")

    def finish_explore(self):
        if self.service is None:
            return

        if not self.is_conn_alive():
            return

        try:
            svc = self.service  # freeze netref once

            # --- remote phase ---
            svc.finish_explore(log=True)
            prob_obs_tensor, prob_pi_tensor, prob_z_tensor, prob_counts = \
                to_local(svc.weighted_dataset())

        except (EOFError, BrokenPipeError):
            # worker already dead — don't crash whole run
            return

        finally:
            # ALWAYS end the worker for this exploration
            self.stop("Worker's shift ended.")

        # --- local phase only ---
        temp = []
        for obs, pi, z in zip(prob_obs_tensor, prob_pi_tensor, prob_z_tensor):
            obs = tuple(obs)
            pi = tuple(pi)
            z = tuple(z)
            temp.append((obs, (pi, z)))

        self.rb.update(temp)

    def flatten_obs_pi_z(self, rich_obs_pi_z):
        cstates, rich_pi_z = zip(*rich_obs_pi_z)  # each entry is (cstate, (pi, z))
        obs_tensor = np.stack([s for s in cstates], axis=0)

        pi_list = []
        z_list = []
        for pi, z in rich_pi_z:
            pi_list.append(pi)  # already a distribution over actions
            z_list.append(z)  # scalar outcome
        pi_tensor = np.array(pi_list, dtype=float)
        z_tensor = np.array(z_list, dtype=float).reshape(-1, 1)

        return obs_tensor, pi_tensor, z_tensor

    def weighted_dataset(self):
        rich_obs_qvs_zs, counts = self.rb.get_full_dataset()
        assert len(rich_obs_qvs_zs) > 0, "Empty replay %s" % (self.rb,)
        counts = np.asarray(counts, dtype='float32')
        # obs_tensor, pi_tensor = self.flatten_obs_qvs(rich_obs_qvs_zs)
        obs_tensor, pi_tensor, z_tensor = self.flatten_obs_pi_z(rich_obs_qvs_zs)
        return obs_tensor, pi_tensor, z_tensor, counts

    def dataset_is_empty(self):
        return len(self.rb) == 0

    def get_replay_size(self):
        return len(self.rb)

    def trim_replay(self):
        self.rb.remove_oldest()

    def next_instance(self, curr_weights):
        if not self.is_first_instance:
            root = self._reconnect()
        else:
            root = self.service
        try:
            try:
                self.curr_prob_name = root.initialise()
            except AssertionError:
                self.curr_prob_name = root.get_problem_name()
            estimator_config = self.enhsp_config if self.enhsp_config is not None else "hadd-gbfs"
            try:
                root.initialise_estimator(estimator_config)
            except AssertionError:
                pass
            self.prob_meta = to_local(self.prob_meta)
            root.make_network(
                self.weights_manager.export_numpy(),
                self.prob_meta,
                self.network_dropout,
                self.network_debug,
                self.network_policy_only,
            )
            print('Made network, starting exploration')
            output = root.explore_from_init_state(curr_weights)

            # self.stop("Worker's shift ended.") TODO: never ever put this back here, this belongs to "finish_explore"
            self.is_first_instance=False
            return output
        except Exception:
            self.stop()
            raise

        #TODO: this method should re-start the worker with the current network weights,
        # and use problem service's explore_from_init_state