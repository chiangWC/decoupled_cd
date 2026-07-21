# XES3G5M static metadata admission

## Purpose

The current KnoField XES3G5M protocol records 1,624 remapped exercises and
241 remapped leaf concepts, but not the official KC routes. Its manifest says
that it was derived from DFCD's processed XES3G5M data. This outcome-blind
audit determines whether the official XES KC tree can be mapped exactly back
to the current protocol before any representation-signal experiment is run.

The audit does not read response labels. It compares student-item ID
multisets, Q assignments, source preprocessing, and static metadata only.

## Frozen sources

- Official XES3G5M repository commit:
  `b5a35caf223c5c5d558da2fb836087fa6af2bef5`.
- Official Google Drive archive:
  `XES3G5M.tar.gz`, SHA-256
  `62d145bd995248f78726a0b6ab69612cf418e3118d0fe01bfe6f4c61b5072f73`.
- DFCD repository commit:
  `6a3127c7dda4aab3a92077634803b48fc2b3e716`.
- DFCD preprocessing parameters from `run.sh`: seed 0, 2,000 requested
  students, 2,000 requested exercises, top 200 concepts, and minimum 50
  responses.
- Current KnoField standard protocol. Its `data.csv`, `train.csv`, and
  `Q_matrix.csv` hashes are frozen in the executable.

DFCD did not commit its generated `map.pkl`. The mapping was reconstructed by
running the unmodified DFCD preprocessing script with `PYTHONHASHSEED=0`.
The regenerated `TotalData.csv` is byte-identical to the tracked DFCD file.
Concept-column order in regenerated dense `q.csv` is not used because the
upstream script enumerates a Python set. Instead, the current Q column ID is
bound directly to the official first-route leaf text through every mapped
exercise; any inconsistency aborts.

## Exact admission conditions

The source is admitted only if all conditions hold:

1. all frozen hashes and both Git commits match;
2. current and tracked DFCD student-item interaction multisets are identical;
3. the reconstructed question map covers exactly all 1,624 current items;
4. every current item maps to an official question and one current Q concept;
5. all items assigned to the same current concept agree on the normalized
   official first-route leaf text;
6. distinct current concept IDs have distinct official leaf texts;
7. every route node exists in the official KC route map;
8. all 241 current concepts are incident to at least one admitted tree path;
9. the materialized graph has no self-loop or duplicate edge.

The route used for a current item is the same first `kc_routes` entry that
DFCD used when constructing its Q matrix. If different items assigned to one
current concept expose different parent paths but the same leaf, all proven
paths are retained. Internal KCs absent from the current Q matrix remain
auxiliary static nodes; model outputs are still restricted to the 241 current
concepts.

## Output

The audit writes a hash-bound sidecar containing:

- current item to official question ID;
- current concept ID to official leaf label;
- typed directed parent-to-child KC-tree edges;
- labels for auxiliary internal nodes;
- exact identity, coverage, ambiguity, and fingerprint reports.

Passing admission only permits the next train-only pseudo-target signal screen.
It does not claim that KC routes improve prediction, authorize a neural module,
or add XES to a final paper table by itself.

## Process note

The first formal invocation at commit `85bf3d9` stopped at import time because
the executable had not added the repository root to `sys.path`. No input
audit or mapping statistic was emitted. The follow-up fix changes only module
discovery; every source, hash, mapping rule, and admission condition is frozen.

The second invocation at commit `6c818b8` passed frozen input verification but
stopped before materialization because the executable checked for
`student_map` while the unmodified DFCD script names the field `stu_map`.
The correction binds to the original schema and changes no audit statistic.
