# Gate D result: graph-family concept-depletion verification

Date: 2026-07-22  
Branch: `codex/concept-depletion-gate-d-graph-family`  
Preregistration: `2026-07-22-concept-depletion-gate-d-graph-family.md`

## Scope

Gate D tested whether graph cognitive-diagnosis models are systematically more
sensitive than a non-graph model to removing a student's target-concept
responses. It reused the frozen Gate B intervention, audit rows and hashes on
ASSIST09, ASSIST17, MOOCRadar, XES3G5M and EdNet. The source standard test
splits were not opened.

The graph families were ORCDF-NCD, SVGCD, RCD and HyperCD. Official EduCDM
KaNCD-GMF was the non-graph reference. RCD and HyperCD were run from the
read-only PyEdmine source at commit
`11dd20f3d103a44821fee674fa0fa5a2a2a5efa2`; all relevant source files were
hash-checked as clean. Each generated runtime was isolated from that source
tree.

The primary within-model effect was

```text
D_model = log_loss(concept-depleted) - log_loss(random-depleted).
```

The direct graph-specific effect was

```text
I_graph = D_graph - D_EduCDM-KaNCD-GMF.
```

Both used 2,000 student-clustered bootstrap replicates with seed 2024. Bold
within-model damages have a strictly positive 95% CI lower bound.

## Results

| Dataset | ORCDF-NCD | SVGCD | RCD | HyperCD | EduCDM KaNCD-GMF | graph support observed/max | positive interactions observed/max | dataset gate |
|---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| ASSIST09 | **+0.024381** | **+0.027601** | **+0.013537** | **+0.016830** | +0.016548 | 4/4 | 0/0 | No |
| ASSIST17 | +0.004296 | **+0.020483** | stopped | +0.001401 | +0.003836 | 1/2 | 1/2 | No |
| MOOCRadar | +0.001321 | **+0.010423** | OOM | +0.004232 | +0.006950 | 1/2 | 0/1 | No |
| XES3G5M | **+0.013706** | **+0.009381** | not started | **+0.002872** | **+0.018742** | 3/4 | 0/1 | No |
| EdNet | +0.002607 | -0.001268 | stopped | **+0.007368** | -0.000665 | 1/2 | 1/2 | No |

The added HyperCD model supports within-model concept-specific damage on three
datasets:

- ASSIST09: `+0.016830`, 95% CI `[+0.005760, +0.028349]`;
- XES3G5M: `+0.002872`, 95% CI `[+0.000564, +0.005146]`;
- EdNet: `+0.007368`, 95% CI `[+0.001067, +0.013142]`.

It does not support ASSIST17 or MOOCRadar. RCD completed ASSIST09 with damage
`+0.013537`, 95% CI `[+0.002018, +0.024996]`. Its fixed MOOCRadar recipe OOMed
on both arms at the first batch; the official forward attempted a
`[1024, 696, 696]` intermediate and required another 1.85 GiB on a 24 GiB GPU.
No batch or architecture fallback was applied.

Only two graph-vs-non-graph interactions were significantly positive:

- SVGCD on ASSIST17: `+0.016648`, 95% CI
  `[+0.005183, +0.027929]`;
- HyperCD on EdNet: `+0.008032`, 95% CI
  `[+0.000798, +0.015103]`.

They occur on different datasets. ASSIST09 is especially informative: all
four graph models have significant positive within-model damage, but none is
significantly more damaged than KaNCD. XES3G5M has three supporting graph
models, but all three completed graph-vs-KaNCD interactions are non-positive.

## Futility stop

RCD's ASSIST17 arms required about 11 minutes per epoch. Its EdNet arms had not
finished the first epoch after more than 37 minutes while both GPUs remained
fully utilized. At that point the final gate had become mathematically
unreachable, even under the most favorable possible missing RCD results:

- ASSIST17, MOOCRadar and EdNet could reach at most two supporting graph
  models, below the required three;
- XES3G5M could reach at most one positive graph-vs-KaNCD interaction, below
  the required two;
- ASSIST09 was complete and had zero positive interactions.

The unfinished RCD ASSIST17/EdNet jobs were therefore stopped for logical
futility, and XES3G5M was not started. These cells are unresolved rather than
negative observations. The analysis reports upper bounds that grant every
unresolved cell a favorable outcome; the maximum possible admitted-dataset
count remains zero. This deviation from running every preregistered RCD job
cannot change the endpoint in either direction.

## Decision

The observed admitted-dataset count is `0/5`, and the most favorable possible
count after granting all unresolved RCD cells success is also `0/5`. The fixed
requirement was at least three datasets. Gate D therefore rejects the broad
claim that student-local target-concept depletion is a systematic,
graph-family-specific CD problem.

The supported statement is narrower: selected graph models on selected
datasets are sensitive to this intervention. That case-level result cannot be
used as the paper's universal problem premise. H/T slices may remain evaluation
axes, but they do not by themselves justify a graph-specific TKC/UKC story.

## Verification artifacts

- HyperCD summary: `results/problem_gate_d/hypercd_screen/gate_b_summary.json`;
- RCD ASSIST09 summary:
  `results/problem_gate_d/previews/rcd_assist09/gate_b_summary.json`;
- final interactions:
  `results/problem_gate_d/final/gate_d_model_interactions.csv`;
- final decision: `results/problem_gate_d/final/gate_d_summary.json`;
- fixed-recipe OOM logs:
  `results/problem_gate_d/formal/rcd/moocradar/*/train.log`.

Prediction alignment was checked on
`audit_row_id/stu_id/exer_id/label`. Model seed was 42; bootstrap seed was
2024. Unit tests, `compileall` and `git diff --check` were run after the final
analysis.
