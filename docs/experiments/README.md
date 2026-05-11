# Experiment Detail Notes

This directory keeps `docs/model_improvement_plan.md` dense.

Use this directory for experiments whose details are still useful for future decisions. Keep the main plan as the high-signal index and put supporting seed rows, diagnostics, commands, and failure-mode evidence here.

Important caveat: many migrated notes were already compressed before this directory existed. A `local_detail_doc` means the currently available detail was moved out of the main plan; it does not guarantee that older uncompressed discussion has been found in git history.

When old detail matters, look it up explicitly with targeted git archaeology, for example:

```bash
git log -S "实验 62-65" -- docs/model_improvement_plan.md
git show <commit>:docs/model_improvement_plan.md
```

Recommended detail status values for `docs/experiment_index.jsonl`:

- `local_detail_doc`: detailed notes are available in this directory.
- `history_enriched_detail`: the local detail doc includes targeted evidence from older git history.
- `compressed_only`: only the compressed main-plan summary is available.
- `older_history_possible`: older uncompressed notes may exist in git history.

New experiment notes should prefer:

- One short verdict near the top.
- Compact seed/result tables instead of prose-heavy rows.
- Explicit failure-mode tags that match `docs/experiment_index.jsonl`.
- Links to result files or branches when those are more useful than copying long logs.
