---
name: numeric-asnet-session-handoff
description: Generate a compact, ready-to-paste fresh-session handoff prompt for Roee Hersco's Numeric-ASNet thesis work, carrying every current response annotation, canonical evidence/documentation routes, the last verified experiment and cluster state, unresolved decisions, and the requested continuation action. Use when moving this project to a new Codex chat. Do not execute experiments or answer the annotations during the handoff turn.
metadata:
  short-description: Build a fresh Numeric-ASNet session handoff
---

# Numeric-ASNet Session Handoff

Create one self-contained prompt that lets a fresh Codex chat continue the
current Numeric-ASNet thesis task without inheriting the full conversation.
The handoff turn is meta-work only: gather and package context, but do not
inspect the live cluster, mutate files outside the handoff artifact, execute
pending work, answer the annotations, or create the new chat.

## 1. Preserve the actual request

- Read the user's latest request and every response annotation in the current
  message.
- Carry every annotated comment into the generated prompt with enough selected
  text or surrounding context to make it understandable.
- Attach its exact one-based directive to the corresponding task:
  `:codex-annotation{index="N"}`.
- Do not renumber, merge, omit or duplicate annotations. Do not answer them in
  the current session; tell the next session to investigate and answer them.
- Include unannotated requests from the latest message as explicit actions.
- If the user supplies a continuation prompt to reproduce, preserve its intent
  and make it the final action request in the generated prompt.

## 2. Route the next session to canonical context

Reference maintained sources instead of reproducing their contents. Include
only paths relevant to the current request.

For thesis evidence and writing context, route the next session to the
`numeric-asnet-thesis-writer` skill, especially its
`references/source-map.md` and `references/evidence-and-writing-rules.md`.

For live experiment or cluster continuation, route it to:

- the `slurm-cluster-operations` skill;
- the `thesis-experiments-health-check` skill;
- `docs/codex_cluster_workflow_efficiency.md`;
- the canonical SSH/access document resolved from the active evidence
  worktree;
- `experiment_tracking/documentation_index_latest.md`;
- `experiment_tracking/result_csv_provenance_index_latest.csv`;
- the campaign README/CSV files directly implicated by the annotations.

If plots or learning curves may change, mention the
`plot-experiment-learning-curves` skill. Do not instruct the next session to
load unrelated skills.

Resolve the current evidence worktree explicitly with Git rather than assuming
that the main checkout is current. When a canonical file is absent from a
sparse checkout but known to exist in a recorded commit, tell the next session
to read it with `git show <commit>:<path>`.

## 3. Carry only ephemeral state that documentation cannot replace

Summarize the latest verified state needed to resume efficiently:

- snapshot timestamp and timezone;
- job IDs, dependencies and controllers;
- scientific completion counts, not only scheduler states;
- current resources and realistic timing when known;
- final conclusions already established;
- incomplete lower bounds and why they are incomplete;
- submitted, running, held, failed and not-yet-submitted work;
- exact operational blockers and safe recoveries;
- decisions already made, including rejected or held branches;
- authorization boundaries and actions that still require user judgment;
- relevant local commits or uncommitted artifacts.

Label the snapshot as stale and require a fresh preflight. Never present an old
queue snapshot as current. Prefer paths and job IDs over long retellings of
historical work.

## 4. Preserve project-wide scientific invariants

Include only the invariants relevant to the continuation, especially:

- validation-led primary RQs only; exclude terminal-led work;
- MPrime belongs in each applicable canonical RQ family;
- fixed search and progressive widening remain separate;
- incomplete work uses classified counts and `>=` lower bounds;
- scheduler completion is not scientific completion;
- paired effects, confidence intervals, exact tests and canonical Holm
  correction govern final RQ claims;
- selected cases and small pilots support mechanisms or dropped-direction
  explanations, not broad efficacy;
- legacy and corrected KL semantics are not mixed inside one method family;
- existing exact-hash results are reused and only genuinely missing identities
  are recovered;
- dirty worktrees and unrelated user changes are preserved.

Do not copy a fixed historical snapshot into this skill. Derive volatile state
from the current conversation and locally documented artifacts each time.

## 5. Produce one paste-ready prompt

Return a single reusable prompt, preferably in one `standard` writing block.
It must be suitable for pasting into a completely new chat and contain:

1. the workspace and continuation framing;
2. skills and canonical documents to read before acting;
3. relevant scientific and operational invariants;
4. the last verified state, explicitly timestamped and marked stale;
5. each annotation as a concrete next-session task with its inline directive;
6. the user's unannotated requests;
7. a final action request that tells the new session to refresh status, act on
   unambiguous recoveries, report conclusions and changed RQs, and distinguish
   work it can perform from decisions requiring the user;
8. any reviewer/subagent requirement the user requested.

The generated prompt should tell the next session to continue the work, not
merely summarize it. It should also say not to repeat information already
available in a referenced canonical file unless that information is essential
to start safely.

After the writing block, add at most one sentence telling the user to paste it
into a completely new chat. Do not recommend `Continue in` unless the user
explicitly prefers carrying the old conversation history.

## 6. Validate before returning

Check all of the following:

- annotation count equals the number of distinct inline directives;
- directive indices exactly match array order;
- every user question and requested action appears once;
- no annotation has been answered prematurely;
- every live-state claim has a timestamp or is clearly historical;
- canonical paths are used instead of duplicated prose where practical;
- the prompt contains no invented result, job ID, permission or completion;
- the next session can identify what to inspect first, what it may safely do,
  what remains held and what needs user judgment;
- no cluster call or experiment mutation was performed in the handoff turn.
