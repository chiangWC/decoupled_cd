# Experiment Detail Notes

This directory keeps `docs/model_improvement_plan.md` dense.

Use this directory for experiments whose details are still useful for future decisions. Keep the main plan as the high-signal index and put supporting seed rows, diagnostics, commands, and failure-mode evidence here.

## Agent-Facing Contract

The experiment ledger is maintained for AI/agent continuity, not for long-form human reading. Prefer stable, searchable, low-ambiguity records over polished narrative.

For routing and filtering, `docs/experiment_index.jsonl` is the primary structured entry point. A detail note should only add evidence that is useful when an agent needs to decide whether to reuse, revisit, reject, or merge an idea.

Minimum useful information for a detail note:

- experiment id or id range
- branch and key commit when available
- base/mainline being compared against
- exact structural or protocol change
- seed metrics or result deltas when available
- verdict and status
- failure-mode or success tags that match `docs/experiment_index.jsonl`
- result directory, command, or git-history pointer when the local note is compressed

Important caveat: many migrated notes were already compressed before this directory existed. A `local_detail_doc` means the currently available detail was moved out of the main plan; it does not guarantee that older uncompressed discussion has been found in git history.

Incomplete historical recovery is acceptable. Do not spend effort reconstructing every old experiment into a full narrative unless it is a mainline component, a strong candidate, or a failure route that agents are likely to retry. When evidence is incomplete, preserve that fact through `detail_status` instead of smoothing it over.

When old detail matters, look it up explicitly with targeted git archaeology, for example:

```bash
git log -S "实验 62-65" -- docs/model_improvement_plan.md
git show <commit>:docs/model_improvement_plan.md
```

Recommended detail status values for `docs/experiment_index.jsonl`:

- `history_enriched_detail`: the local detail doc includes targeted evidence from older git history.
- `local_detail_doc`: detailed notes are available in this directory, but no extra git-history recovery has been done.
- `migrated_summary`: this is only a short summary migrated from the compressed main plan; do not treat it as full detail.
- `compressed_only`: only the compressed archive summary is available.
- `older_history_possible`: older uncompressed notes may exist in git history.

New experiment notes should prefer:

- One short verdict near the top.
- Compact seed/result tables instead of prose-heavy rows.
- Explicit failure-mode tags that match `docs/experiment_index.jsonl`.
- Links to result files or branches when those are more useful than copying long logs.
- A clear note when the evidence is only a partial recovery from compressed history.
