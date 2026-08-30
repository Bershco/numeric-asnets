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

For unattended checks use the alias directly, for example:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=20 uni-cluster hostname
```

If that command fails before authentication, verify `whoami`, `$env:USERPROFILE`,
and the resolved SSH executable before blaming the VPN. Only treat the VPN as
the likely cause after three attempts from the documented profile fail to route.
