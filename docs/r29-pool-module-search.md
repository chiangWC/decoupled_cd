# r29 Pool-Driven Attributable Module Search

## Provenance and boundary

r28 was frozen at commit `c35d275` before this branch was created. r29 reuses only
the common data, training, evaluation and audit harness. The generic r28
`difficulty_capacity` control is available as `marginal_anchor`, a performance
floor that is explicitly not a paper contribution.

The framework has two boxes: one replaceable State Completion Module and a fixed
Q-conditioned pooled NCF Diagnosis. Diagnosis receives `framework_state` as its
only student-specific input. There is no student-ID embedding, legacy state
path, residual prediction head, dataset router or per-dataset topology.

The non-contribution marginal anchor was rerun under the r29 interface on
MOOCRadar validation. Its S/H/T AUC is
`0.930650/0.926934/0.935917`, with external margins
`+0.001097/+0.002755/+0.002808`; this is a strict win and confirms that the
fixed Diagnosis and training harness can reach the live external line.

## Pool admission

The model seed is fixed to 42 and the data split seed to 2024.

- Junyi uses the verified current KnoField rows. The rebuilt standard and
  holdout variants add stable row identities and reproduce the original row
  partitions. Its 706 exercises map one-to-one to 706 concepts, so exact-zero
  means both unseen item and unseen concept. Holdout validation/test contain
  35,913/70,687 eligible exact-zero rows.
- `ednet_clean_v2` combines the three contaminated source splits, removes 70,765
  exact duplicate quadruples, retains 42,679 student-exercise groups with
  conflicting repeated labels, and atomically assigns every `(student,
  exercise)` group by a stable seed-2024 hash. It has 753,564 clean rows, 11,988
  exercises and 189 concepts. Standard and holdout cross-split group overlaps
  are zero; holdout validation/test contain 16,366/32,706 eligible exact-zero
  rows.

Both datasets remain `external_reproduction_pending` until seed-42 ORCDF and
SVGCD predictions are trained and verified row-by-row. Historical Junyi numbers
are sanity checks only.

## Ordered mechanism decisions

### Response-conditioned Peer Completion — rejected before implementation

The train-only pseudo-holdout compares response-similarity peers against a
coverage-only peer control using the fixed `retrieval_pool=256`, `peer_k=32`,
`min_overlap=3` recipe. Junyi improved by only `+0.000532` AUC and `-0.000280`
Brier; EdNet improved by `+0.000843` AUC and `-0.000083` Brier. Neither reached
the preregistered `+0.002/-0.001` gate, so the Coral-inspired direction was not
implemented as a framework module.

### Accuracy-oriented Wasserstein Flow — rejected before implementation

The four-step, 20%-masked train-cell proxy used an exact-parameter deterministic
control and a shared initialization. Flow minus control AUC/Brier was:

| Dataset | Delta AUC | Delta Brier |
|---|---:|---:|
| MOOCRadar | -0.009198 | +0.004419 |
| Junyi | -0.004026 | +0.001848 |
| EdNet | -0.007507 | +0.001553 |

The NewImp-inspired direction therefore failed on every audited dataset and was
not promoted to the framework.

### Meta-adapted Implicit State Function — rejected by clean control

The candidate initializes a student context from train-only history and performs
exactly three inner reconstruction updates before evaluating a shared implicit
student-concept state function. It has no free student parameter or low-rank
mastery factor. Its Capacity Control instantiates the exact same parameters and
uses deterministic amortized context updates instead of inner gradients. The
Direct control uses observed evidence projection plus a static prior.

Full and Capacity have identical parameter counts (196,442) and initialization
hashes. On the first formal MOOCRadar screen, Full minus Capacity S/H/T was
`-0.000574/-0.000455/-0.000441`; the student-clustered paired-bootstrap 95% CI
for delta T was `[-0.000703, -0.000168]`. Full also missed every external axis.
Because retaining the MOO ordinary win was a necessary first-screen condition,
the two-dataset gate became unreachable and the candidate was rejected without
using Junyi as a post-hoc rescue dataset.

### Post-r29 response-distribution completion — rejected before implementation

After all three ordered r29 candidates failed, the next literature question was
whether the strong marginal anchor discarded useful shape information in each
student's response/item-difficulty distribution. A fixed 32-dimensional random
Fourier representation was compared with a same-dimensional degenerate
marginal representation; both used the same downstream logistic model. The
20% pseudo-holdout came only from training rows, and fit/evaluation students
were disjoint.

| Dataset | Delta AUC | Delta Brier |
|---|---:|---:|
| MOOCRadar | -0.006671 | +0.002030 |
| Junyi | +0.000763 | -0.000273 |
| EdNet | +0.000059 | +0.000025 |

No dataset reached the fixed `+0.002` AUC activation threshold, and MOO moved
materially in the wrong direction. The distribution-regression direction was
therefore rejected at proxy level and was not turned into a framework module.
The design was informed by distributional-input and kernel-embedding work, but
no source model or code was adopted.

## Commands

```bash
conda activate decoupled_cd

python -m unittest discover -s tests -p 'test_r29*.py' -v

python scripts/run_r29_campaign.py \
  --dataset moocradar --dataset junyi \
  --data-root /home/xph/jwc/research/knofield_data \
  --output-root results/r29/meta_screen \
  --gpus 0,1,3

python scripts/evaluate_r29_campaign.py \
  --results-root results/r29/meta_screen \
  --dataset moocradar --dataset junyi \
  --output results/r29/meta_screen_report.json
```

The scheduler uses live free memory and utilization; an occupied GPU remains
eligible when the declared peak fits. OOM is a failed task and never changes the
registered recipe silently.
