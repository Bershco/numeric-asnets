import collections
import dataclasses
import os
import types
from multiprocessing import Process

from rpyc import BaseNetref
from rpyc.utils.classic import obtain
import tensorflow as tf

def to_local(obj):
    """Convert a NetRef to an object to something that's DEFINITELY local."""
    # can probably smarter here (e.g. not copying netrefs, using joblib for
    # efficient Numpy support); oh well
    # TODO: try using encode/decode with joblib instead! Could be much, much
    # faster.
    # TODO: make sure that you're transmitting observations as byte tensors
    # whenever possible (or at most float32s).
    # return deepcopy(obj)
    # return obtain(obj)
    # === First: RPyC proxy check ===
    # (must come *before* container checks, because some proxies act iterable)
    if isinstance(obj, BaseNetref):
        try:
            return obtain(obj)
        except Exception:
            return obj  # fallback if obtain() fails

    # === Primitive / simple immutable types ===
    if obj is None or isinstance(obj, (str, bytes, int, float, bool, complex)):
        return obj

    # === TensorFlow objects: must NOT obtain ===
    try:
        if isinstance(obj, (tf.Variable, tf.Tensor, tf.Module, tf.keras.layers.Layer)):
            return obj
    except Exception:
        pass  # don't break if tf isn't loaded yet

    # === Containers ===
    if isinstance(obj, dict):
        return {to_local(k): to_local(v) for k, v in obj.items()}

    if isinstance(obj, collections.abc.Mapping):
        return type(obj)((to_local(k), to_local(v)) for k, v in obj.items())

    if isinstance(obj, (list, tuple, set, frozenset)):
        seq = (to_local(v) for v in obj)
        return type(obj)(seq)

    # === Dataclasses ===
    if dataclasses.is_dataclass(obj):
        field_values = {f.name: to_local(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return type(obj)(**field_values)

    # === Namedtuples ===
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(to_local(v) for v in obj))

    # === Modules / functions / types ===
    if isinstance(obj, (types.ModuleType, types.FunctionType, type)):
        return obj

    # === Default fallback ===
    return obj


def _shutdown_proc(proc: Process):
    if proc is None:
        print(f"[DEBUG] proc is {proc}")
        return

    # Only the creating process may touch it
    if proc._parent_pid != os.getpid():
        print(f"[DEBUG] Current process {os.getpid()} is not proc's parent {proc}")
        return

    if proc.is_alive():
        print(f"[DEBUG] Current process {os.getpid()} is terminating {proc}")
        proc.terminate()
        proc.join(timeout=10)

def find_netrefs(obj, path="root", seen=None, limit=20):
    if seen is None: seen=set()
    if id(obj) in seen: return []
    seen.add(id(obj))
    out = []
    if isinstance(obj, BaseNetref):
        out.append((path, type(obj)))
        return out
    # common containers
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:200]:
            out += find_netrefs(k, path + ".<key>", seen, limit)
            out += find_netrefs(v, path + f"[{repr(k)[:40]}]", seen, limit)
            if len(out) >= limit: return out
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(list(obj)[:200]):
            out += find_netrefs(v, path + f"[{i}]", seen, limit)
            if len(out) >= limit: return out
    else:
        # scan attributes conservatively
        for name in getattr(obj, "__dict__", {}).keys():
            if name.startswith("_"):  # skip noisy internals
                continue
            try:
                out += find_netrefs(getattr(obj, name), path + "." + name, seen, limit)
            except Exception:
                pass
            if len(out) >= limit: return out
    return out

def find_netrefs_all(obj, path="root", seen=None, limit=50):
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return []
    seen.add(oid)

    out = []
    if isinstance(obj, BaseNetref):
        out.append((path, type(obj), repr(obj)[:200]))
        return out

    # containers
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:500]:
            out += find_netrefs_all(k, path + ".[key]", seen, limit)
            out += find_netrefs_all(v, path + f"[{repr(k)[:80]}]", seen, limit)
            if len(out) >= limit:
                return out
        return out

    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(list(obj)[:500]):
            out += find_netrefs_all(v, path + f"[{i}]", seen, limit)
            if len(out) >= limit:
                return out
        return out

    # scan ALL attributes, including private
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for name, val in list(d.items()):
            try:
                out += find_netrefs_all(val, path + "." + name, seen, limit)
            except Exception:
                pass
            if len(out) >= limit:
                return out

    return out