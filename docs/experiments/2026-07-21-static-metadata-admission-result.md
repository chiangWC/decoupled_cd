# Static side-information admission result

## Frozen execution

- Formal code commit: `e4a446fbbc8a4be37ab1ee328c8c7c6c6779f061`.
- Audit payload SHA-256:
  `c034c24f2e9f0cde2ee5ff8a0ed13c47767c950fa93ac6a4ccd23770fc0bc2af`.
- All 23 input hashes and the three Junyi Git blobs matched their frozen
  values.
- Identity used `data.csv`; reachability used train/validation/Q only. No
  model result, prediction, test row, AUC, or checkpoint was opened.
- Fixed gate: at most four hops, ASSIST group size at most 10% of items,
  incremental target-row reachability at least 25%, and at least 3/4 datasets
  passing.

## Result

The route did **not** activate: only Junyi passed, so the result is 1/4 rather
than the required 3/4.

| Dataset | Identity | Target rows | Metadata-path rows | Q-only rows | Incremental rows | Incremental fraction | Gate |
|---|:---:|---:|---:|---:|---:|---:|:---:|
| ASSIST09 | exact | 10,398 | 800 | 2,871 | 468 | 4.5009% | fail |
| ASSIST17 | exact | 6,851 | 511 | 6,431 | 27 | 0.3941% | fail |
| NIPS34 | exact | 8,464 | 8,464 | 8,464 | 0 | 0.0000% | fail |
| Junyi | exact | 35,913 | 34,068 | 0 | 34,068 | 94.8626% | pass |

The incremental column is the set of target rows for which at least one
missing concept is reachable through a metadata-containing path but is not
reachable in the physically separate Q-only graph under the same hop budget.
It is the only reachability quantity used by the gate.

## Interpretation

- ASSIST09 templates/assistment groups and invariant ASSIST17 problem types
  add little concept access beyond Q under this protocol.
- The NIPS hierarchy produces paths for every target, but Q alone already
  reaches every one of them within four hops. Its apparent 100% metadata-path
  coverage is therefore not attributable to the hierarchy.
- Junyi's prerequisite/similarity graph provides substantial genuinely new
  topology, but a one-dataset mechanism cannot support the required common
  three-dataset architecture.

This audit measures only whether the side information can reach otherwise
unreachable target concepts. It does not claim predictive usefulness, and the
negative route decision means the planned real-vs-shuffle predictive audit is
not run.

## Decision

Reject **Typed Curriculum Path Completion** as the next shared framework
module. Do not implement, tune, rename, or run a neural version of this route.
The Junyi result may be retained as a dataset-specific observation, not as a
common paper module.

The next module search must use information that is present in at least three
winning-candidate datasets. Static curriculum metadata may be used only if a
future, independently sourced data pool changes this admission result; it
must not be attached to the current model as an optional dataset-specific
branch.

## Later scope correction

The rejection above applies to the preregistered claim that metadata must add
otherwise unreachable topology on at least three of these four protocols. It
must not be read as evidence that real static relations lack predictive value,
or that metadata-rich datasets cannot support a publishable component.

A separately preregistered train-only signal audit subsequently compared real
relations with Q-only and degree/type-matched rewires. It found a sizeable
ASSIST09 pseudo-overall AUC gain (+0.018898) even though the exact-zero gain did
not pass the completion gate. Thus the original topology result remains
correct, while its earlier broad wording about the whole research direction is
superseded. See
`2026-07-21-static-metadata-signal-result.md`.

## Process deviations

Before the formal run, an EdNet schema preview displayed five response labels;
none entered this audit. Code review also corrected a weaker
`used-metadata-path` definition after provisional topology counts had been
seen; the numerical thresholds were unchanged and the correction made the
gate stricter.

The first formal invocation at `9a672a3` aborted before emitting a payload
because the Junyi provenance assertion assumed a scalar code instead of a
list. Commit `e4a446f` fixed only that schema handling and documented the
failure; sources, hashes, thresholds, and decision rules did not change.
