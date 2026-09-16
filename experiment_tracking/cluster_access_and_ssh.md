# Cluster access and SSH

For every Codex cluster or experiment-status task, also read
`../docs/codex_cluster_workflow_efficiency.md`. This file governs connection,
profile, VPN and retry behavior; the efficiency document governs batching,
asynchronous Slurm execution and status-report cadence.

Use only the `uni-cluster` SSH profile from `C:\Users\roeeh\.ssh\config`.

- Windows account: `roee-mobile-pc\roeeh`
- Windows profile: `C:\Users\roeeh`
- SSH alias: `uni-cluster`
- Cluster account: `hersco`
- Alias target: `slurm.bgu.ac.il`
- Identity: `~/.ssh/id_ed25519_uni_cluster`

Do not substitute a guessed alias such as `bgu-cluster`, a direct login-node
hostname, or a sandbox/system Windows profile. Those paths caused recurring
host-key, missing-key, and misleading VPN diagnoses.

For Codex desktop work, a sandboxed shell may expose a different Windows
profile and therefore fail to resolve `uni-cluster` even while the VPN is
healthy. Such a failure is not a cluster or VPN observation. The connection
must be run through the normal `C:\Users\roeeh` profile with
`C:\Windows\System32\OpenSSH\ssh.exe`; verify that the alias resolves to user
`hersco` at `slurm.bgu.ac.il`. This is the exclusive supported route for these
experiment checks.

For unattended checks use the alias directly, for example:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=20 uni-cluster hostname
```

If that command fails before authentication, verify `whoami`, `$env:USERPROFILE`,
and the resolved SSH executable before blaming the VPN. Only treat the VPN as
the likely cause after three attempts from the documented profile fail to route.

## Mandatory retry procedure

After **every** failed SSH attempt, wait at least one minute before retrying.
Use that minute to reread this file and verify all of the following before the
next attempt:

- Windows account is `roee-mobile-pc\roeeh`.
- Windows profile is `C:\Users\roeeh`.
- Executable is `C:\Windows\System32\OpenSSH\ssh.exe`.
- Alias is exactly `uni-cluster`.
- The alias resolves to cluster user `hersco` at `slurm.bgu.ac.il`.

Do not spend that minute repeatedly retrying, guessing another alias, using a
direct login-node hostname, or diagnosing the VPN from a sandbox-profile
failure. Only after three correctly spaced failures through this exact route
should the VPN be reported as the likely blocker.

## After a suspected cluster-wide outage

A restored SSH route does not prove that Slurm, compute nodes, accounting, or
shared storage recovered cleanly. Before releasing or submitting anything,
follow `cluster_outage_recovery_plan.md` and run the read-only
`scripts/post_outage_cluster_audit.py` inventory. The frozen pre-outage baseline
is `pre_outage_job_inventory_20260901_1037.csv`. Never blanket-resubmit all jobs:
some may have completed before shutdown, some may retain resumable checkpoints
or per-instance completion records, and deliberate scientific holds must remain
held.
