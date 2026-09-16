# Codex cluster workflow efficiency

This document defines the default execution pattern for Codex tasks involving
SSH, Slurm, experiment deployment, monitoring, and cluster-backed
documentation updates. Its purpose is to reduce wall-clock time and avoid
unnecessary scheduler and SSH round trips without weakening safety or
scientific verification.

Read this together with
`../experiment_tracking/cluster_access_and_ssh.md`. The access document is the
authority for Windows profile, SSH alias, VPN diagnosis and retry behavior;
this document is the authority for efficient scheduler and status workflows.

## Default cluster workflow

1. Perform one batched read-only preflight.
2. When the requested actions are already approved and safely compatible,
   perform deployment, cancellation, repair, or submission in one carefully
   validated mutation operation.
3. Perform one immediate post-operation verification.
4. Continue all independent local analysis, documentation, plotting, and code
   work while cluster jobs run.
5. Poll once after that independent work, or immediately before the final
   response.
6. If jobs remain active, report their state, resources, dependencies, and hard
   time bounds. Do not wait idly or repeatedly poll unchanged state.

Use additional checks only after a failure, ambiguity, dependency problem, or
material safety concern.

## SSH and scheduler round trips

- Batch related read-only checks into one SSH call where practical.
- Batch related approved cluster mutations into one SSH call where doing so
  remains understandable, auditable, and safe.
- Avoid separate connections for information that can be retrieved together.
- Do not inspect unchanged scheduler state multiple times in one turn.
- A successful cancellation does not require waiting for every task to vanish
  before continuing unless its continued presence creates a real conflict.

## Asynchronous Slurm execution

- Treat Slurm dependency chains as asynchronous workflows.
- Submit the complete dependency-gated chain, verify that every job entered the
  intended state, and then continue independent work.
- Use `afterok`, `afterany`, or another appropriate dependency to let Slurm
  remember when downstream work may start.
- Do not wait for job completion unless the result is required to answer the
  current request.
- A pending or running job is a valid end-of-turn state when its submission and
  dependencies have been verified.

For smoke-gated work, prefer:

```text
smoke -> afterok scientific work -> afterok materialization/evaluation
```

While the smoke runs, perform independent local work. Query it once near the
end of the turn. If it is still active, report that fact instead of repeatedly
polling.

## Scope control

- Do not broaden a focused request into a complete repository, documentation,
  or experiment audit unless the user explicitly requests that wider scope.
- Do not silently add optional follow-up work to the current completion
  criteria.
- If optional work would delay the requested answer, report it as a next step.
- Keep documentation changes limited to the requested experiment and directly
  affected canonical status/provenance records.
- Prefer a prompt, accurate partial-status report over delaying the response to
  obtain results that are not yet required.

## Verification proportionality

- Verify exact targets before destructive or externally visible actions.
- Run tests and checks proportional to the scientific and operational risk.
- Once the relevant checks pass, do not repeat them without new evidence of a
  problem.
- Use one preflight, one mutation operation, and one verification by default.
- A failed attempt justifies a targeted repair and one renewed verification;
  it does not automatically justify a broad re-audit.

## Reporting requirements

For submitted or live work, report:

- job IDs and task count;
- running, pending, completed, or failed state;
- requested CPUs and memory;
- dependency structure;
- expected duration when defensibly estimable;
- scheduler hard bound;
- output and provenance location;
- whether the current response depends on completion.

Do not imply that a scheduler hard limit is an expected runtime.

## Short instruction form

When context must remain compact, use this equivalent instruction:

> For Slurm work, default to one preflight, one batched mutation, one immediate
> verification, asynchronous dependency-based execution, productive local work
> while jobs run, and at most one final status poll. Do not broaden the task or
> wait for completion unless the requested answer requires it.

## Related canonical instructions

- `../experiment_tracking/cluster_access_and_ssh.md`: connection profile,
  bounded retry procedure, VPN diagnosis and post-outage access rules.
- `../experiment_tracking/README.md`: canonical experiment and result sources,
  provenance requirements and reporting contracts.
- `../experiment_tracking/cluster_outage_recovery_plan.md`: read-only recovery
  workflow after a suspected cluster-wide outage.

## Instruction precedence

Explicit user instructions for the current request take precedence over this
default. Safety requirements and exact-target verification still apply.
