# Static side-information admission audit

## Decision being made

This is an outcome-blind activation audit, not a model experiment. It decides
whether public, static curriculum metadata is sufficiently aligned with the
current KnoField protocols to justify a new TKC-to-UKC completion candidate.
Response outcomes are never extracted, inspected, or used by audit logic or a
statistic. Whole source/protocol files are covered by pre-frozen cryptographic
hashes; the frozen Junyi JSON is decoded only to select student/item ID keys,
so unrelated fields may be deserialized but are never accessed. No model
prediction, AUC, checkpoint, test ID, or test metric is opened. Identity uses
`data.csv`; reachability and activation use training IDs, validation IDs, and
Q only.

If the gate passes, the next experiment may pre-register a complete **Typed
Curriculum Path Completion** module. That module would use train-history
responses only as student boundary conditions and use typed static paths to
complete UKC states. The present audit does not select model parameters or
claim that such a module works.

## Frozen sources and allowed fields

The executable receives all paths explicitly; no machine-specific path is
embedded in source code.

| Dataset | Identity proof | Allowed static relation | Explicitly excluded |
|---|---|---|---|
| ASSIST09 | Reproduce the public preprocessing using only user, item, and skill IDs; require the expanded `(student,item,concept)` multiset and Q pairs to equal the current protocol | invariant `template_id` and `assistment_id` memberships | correct, hint/attempt/time/opportunity, answer text, teacher/class/school, sequence because it is not item-invariant, and coarse answer type |
| ASSIST17 | Reproduce the published first-attempt preprocessing using only IDs, skill and timestamp; require the `(student,item,concept)` multiset and Q pairs to equal the current protocol | `problemType` only for items having one invariant value across the source | multi-valued `problemType`, correct and all behavioral/affective/student fields; assignment and assistment IDs because they vary with delivery context |
| NIPS34 | Require current `exer_id == QuestionId`; recover the unique sorted raw-SubjectId encoding that exactly reconstructs every current Q row | `ParentId`/`Level` subject hierarchy | answer metadata, correctness, image content, and text/name embeddings |
| Junyi | Require the current interaction-ID multiset to equal the official RCD `log_data.json`; require its SHA-256 to match the tracked official copy | tracked RCD `K_Directed` prerequisite and `K_Undirected` similarity relations, induced on current protocol concepts | scores, response order, graph construction from current responses, and raw-name mappings that cannot be proven |

The only accepted protocol profiles are `assist_09_chold_v2`,
`assist_17_chold_v2`, `nips34_chold_v2`, and `junyi_chold_v2`. Their
`data/train/valid/Q` SHA-256 values and every source-file SHA-256 are immutable
constants in `scripts/audit_static_metadata_admission.py`; a mismatch aborts.
The four hop/count/fraction thresholds are not CLI options.

Junyi is frozen to RCD commit
`178e6a9d1485265cbc5f2f5dc93ecbe5f3c28ef0`; the executable compares each
working source with the Git blob at that commit and its frozen SHA-256. Graph
pairs are deduplicated and self-loops removed. Auxiliary Junyi concepts absent
from the current protocol are excluded. The graph data is used subject to the
Junyi non-commercial data terms and requires the dataset and RCD citations; no
RCD model code is copied.

## Graph and reachability protocol

The audit builds an undirected reachability view of the relations because a
future typed module would explicitly add inverse relation types. Q edges are
marked as base edges; every admitted metadata edge is marked separately.

For each validation student, the sources are only that student's training
items. With the same four-hop budget the audit computes:

- R_meta: missing concepts reached by a path containing a metadata edge;
- R_Q: missing concepts reached in a physically separate Q-only graph;
- incremental reachability: R_meta minus R_Q.

Only incremental reachability enters the gate. Thus neither a direct Q path
nor a redundant metadata path can pass. The target scope is:

- exact-zero for ASSIST09, ASSIST17, and Junyi;
- train-history coverage below 0.5 for NIPS34, matching the live registry.

For low-coverage rows, reachability is computed only for the missing target
concepts. The audit reports both “any missing concept reachable” and “all
missing concepts reachable,” plus shortest-path counts.

Metadata category values incident to one item are useless and values incident
to more than 10% of mapped items are too coarse; both are excluded before
reachability is measured. This rule applies to ASSIST categorical metadata,
not to expert prerequisite/taxonomy edges.

## Fixed gate

A dataset activates only if all conditions hold:

1. its source-to-protocol identity check is exact;
2. at least 99% of current items have a proven source mapping;
3. the validation target contains at least 100 rows;
4. at least 25% of target rows have one or more missing target concepts
   incrementally reachable over the same-hop Q-only graph.

The route activates only if at least three of the four datasets pass. The
thresholds cannot be changed after reading the audit output.

Passing this gate permits only a second, train-only pseudo-holdout audit. That
audit must compare real relations against a degree- and relation-count-matched
shuffle on at least two datasets before any neural module is implemented.
Failing this gate rejects static typed-path completion without opening model
validation or test results.

## Pre-registration process note

During exploratory EdNet provenance inspection before this document was
frozen, a generic five-row schema preview displayed the response-label column.
Those values were not used in any mapping, threshold, or admission decision,
and EdNet is not part of this four-dataset audit. This deviation prevents a
claim that every earlier exploratory command was label-blind; the executable
audit and its fixed gate remain outcome-blind as defined above.

Before the formal run, code review found that the first draft counted any path
containing metadata, even when the same target was already Q-reachable. Some
provisional topology counts had therefore been seen. The implementation was
corrected to the stricter incremental definition above while retaining the
already written 4-hop, 25%, and 3-of-4 thresholds. The corrected source and
this amendment must be committed before the formal run. Consequently this is
an outcome-blind, validation-topology audit, not a claim of blind topology
testing or predictive effectiveness.
