# Cluster access and SSH

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
