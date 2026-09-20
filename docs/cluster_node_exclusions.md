# Cluster node exclusions

This is the canonical, versioned exclusion list for thesis Slurm submissions.
Load it together with `codex_cluster_workflow_efficiency.md` and
`../experiment_tracking/cluster_access_and_ssh.md` before creating or
repairing cluster work.

## Hard operational exclusions — 20 September 2026

Use this compact Slurm form:

```text
ise-cpu-intl-[01,05,15,18,24,26]
```

These nodes repeatedly failed or requeued jobs within seconds or a few minutes
without producing a scientific result. The evidence window is 19--20 September
2026:

| Node | Quick failed/requeued/cancelled attempts | Successful controls in the same audit | Reason for hard exclusion |
|---|---:|---:|---|
| `ise-cpu-intl-01` | 15 | 0 | repeated pre-application operational failure |
| `ise-cpu-intl-05` | 12 | 0 | repeated 2:01 cancellation/requeue pattern |
| `ise-cpu-intl-15` | 28 | 0 | repeated pre-application operational failure |
| `ise-cpu-intl-18` | 20 of 21 audited attempts | 0 | repeated 2:01 requeues without stdout |
| `ise-cpu-intl-24` | 75 of 76 audited attempts | 0 | repeated quick operational failure |
| `ise-cpu-intl-26` | 33 | 0 | repeated signal/pre-application failure across policy and V1 work |

This list intentionally excludes nodes whose failures are attributable to job
memory requests, user quota, or another workload-specific cause. In particular,
`ise-cpu128-04`, `ise-cpu-intl-03`, `ise-cpu-intl-07`,
`ise-cpu-intl-22`, and `ise-cpu-intl-23` have successful controls and are not
global exclusions.

## Maintenance rule

- Add a node only after repeated short operational failures across attempts and
  no convincing successful control in the same evidence window.
- Do not globally exclude a node because a job OOMed, exhausted user quota,
  timed out scientifically, or requested unsuitable resources.
- Keep workload-specific exclusions in the experiment manifest; do not promote
  them here without cross-workload evidence.
- Re-audit this list after cluster maintenance or after a node demonstrates
  repeated successful controls. Historical exclusions are not permanent facts.
- Record the exact exclusion string in every submitted manifest or controller
  so recovery jobs cannot silently re-enter known-bad nodes.
