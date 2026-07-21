# XES3G5M static metadata admission result

## Frozen execution

- Formal code commit:
  `3db9ab05c47647c6b45c6650fa5010e592eb123a`.
- Audit payload SHA-256:
  `1c896e30d410e92deaab803eb5ad7cf420ef667ddcb3a61839577e18974bf2ec`.
- Official XES repository commit:
  `b5a35caf223c5c5d558da2fb836087fa6af2bef5`.
- DFCD repository commit:
  `6a3127c7dda4aab3a92077634803b48fc2b3e716`.
- No response-label column was read by the formal admission executable.

## Result

XES official static metadata is admitted for the current KnoField protocol.

| Check | Result |
|---|---:|
| Current/DFCD student-item multiset | exact, 207,204 rows |
| Current/reconstructed item map | 1,624 / 1,624 |
| Current concept to official leaf map | 241 / 241 |
| Unique official leaf labels | 241 |
| Current concepts with multiple proven parent paths | 65 |
| Maximum proven paths per current concept | 3 |
| Materialized tree nodes | 423 |
| Auxiliary internal KC nodes | 182 |
| Directed parent-child edges | 457 |
| Current concepts incident to tree | 241 / 241 |
| Self-loops / duplicate edges | 0 / 0 |

DFCD's regenerated `TotalData.csv` matched its tracked file byte for byte.
The dense regenerated `q.csv` did not preserve the tracked concept-column
order because the upstream script enumerates a Python set. The audit therefore
did not trust that column order: every current concept ID was independently
bound to the official first-route leaf through all current items, and all
items agreed without conflict.

## Decision

The official XES tree may enter a train-only predictive-signal experiment and,
if that passes, a curriculum-relation representation component. Internal KCs
may be used as auxiliary nodes, but the model's diagnosed concepts remain the
241 concepts in the current Q matrix.

Admission is provenance only. It does not establish a predictive benefit,
external win, or paper-module contribution.
