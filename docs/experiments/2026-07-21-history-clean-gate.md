# Clean History Gate after Requirement Rejection

## Decision question

The Q-consistent Requirement gate rejected joint nonlinear Q-item composition.
The rejected 14-job Requirement full factorial is therefore retired. This last
History screen asks one isolated question:

> With the same information-matched factorized item Requirement on both paths,
> does attempted-item semantic History materially outperform a calibrated
> summary of accuracy, attempted-item difficulty, confidence, and coverage?

The matched History-Full anchor is not the previously named whole-model Full.
For every dataset it is the existing validation-only
`<dataset>_holdout_wo_requirement` artifact:

```text
History-Full anchor:
  calibrated_history + factorized_item_control

History control (new wo_both job):
  calibrated_summary_control + factorized_item_control
```

The two paths preserve the same target item ID, Q view, factorized item
representation, pooled NCF diagnosis, data order, mask, optimizer recipe and
seed 42. Only the History representation changes.

## Four validation jobs

`history_gate` contains exactly four holdout-validation tasks:

| Dataset | Split | Variant |
|---|---|---|
| ASSIST17 | holdout | `wo_both` |
| MOOCRadar | holdout | `wo_both` |
| XES3G5M | holdout | `wo_both` |
| Junyi | holdout | `wo_both` |

No standard or test task is launched. `valid.csv` is passed as the training
harness test placeholder and `--evaluation-stage validation` leaves
`test_metrics=null`. Before execution, the runner hashes and verifies all four
History-Full summaries, checkpoints and row-level predictions. It rejects an
anchor that used test metrics, a different seed, a different History or
Requirement mode, or a different architecture/ablation fingerprint.

The runner also refuses formal execution unless the worktree is clean, HEAD
matches `--expected-commit`, and a fetched remote ref contains that HEAD.

## Pre-registered decision rule

For each dataset, define

```text
Delta T = T AUC(calibrated_history + factorized_item_control)
        - T AUC(calibrated_summary_control + factorized_item_control).
```

Only datasets on which the History-Full anchor retains the external win are
eligible. Following `docs/research_goal.md`, History passes this screen only if
all conditions hold:

- at least two eligible datasets have `Delta T >= 0.005`;
- at least one eligible dataset has `Delta T >= 0.01`;
- at least one student-clustered paired-bootstrap 95% CI has lower bound `> 0`;
- Full does not regress against the control by more than `0.001` on any
  measured winning H/T axis.

The target mask must be rebuilt from train/valid/Q under the Q-consistent
protocol, and paired bootstrap uses students as clusters. This is not a
multi-seed experiment.

Because this deliberately minimal screen trains no standard control, it can
reject History but cannot by itself certify the terminal S-axis non-regression
condition. If it clears the large target-effect thresholds, the S-axis check
must be resolved before calling History a qualified paper module. A failed
screen rejects the current History mechanism immediately; it must not be
rescued with the retired Requirement factorial.

## Formal validation result

All four jobs completed with exit status 0 at implementation commit
`50f0f4006503e398db6e570a999ba35f4c3862a8`. The History-Full and control
initialization hashes match exactly for every dataset, and all models share
architecture fingerprint `099906acdba8c3b4`. No test artifact was opened.

| Dataset | Full H | Control H | Delta H | Full T | Control T | Delta T | student-clustered Delta T 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASSIST17 | .799879091 | .784089164 | +.015789927 | .781492464 | .775977448 | +.005515016 | `[+.001274223, +.009824270]` |
| MOOCRadar | .926716237 | .926556680 | +.000159557 | .935352052 | .935386386 | -.000034333 | `[-.001171523, +.001147225]` |
| XES3G5M | .787695788 | .786159274 | +.001536514 | .785818227 | .784459719 | +.001358508 | `[-.000270570, +.002988677]` |
| Junyi | .824024689 | .822692322 | +.001332367 | .824024689 | .822692322 | +.001332367 | `[+.000329237, +.002342901]` |

The matched History-Full anchor remains a strict external winner on all four
datasets. However, only ASSIST17 reaches `Delta T >= 0.005`, no dataset reaches
`Delta T >= 0.01`, and only ASSIST17 and Junyi have a positive CI lower bound.
The pre-registered gate is conjunctive, so the effect-size checks fail and the
current History mechanism is rejected.

No standard control is added: the necessary T-effect gate has already failed,
so an S-axis run cannot reverse the decision and would spend compute only to
complete a rejected module's table. The architecture therefore remains a
four-strict-win performance path with zero qualified paper modules.

## Result artifacts

All paths are relative to
`results/goal_two_module/factorized_requirement_factorial_v10/`:

- execution identity: `runner_identity_history_gate.log`;
- locked anchor audit: `locked_history_anchor_audit.json`, SHA-256
  `369e6831e9976f8b4bc7696e01d8bc5143dc7c4d14069a2f4711bdd825a4e1a0`;
- trained controls: `<dataset>_holdout_wo_both.{json,runner.log}` plus
  checkpoints, slice summaries and row-level predictions;
- paired-bootstrap results: `<dataset>_history_gate_bootstrap.json` with
  2,000 student-cluster replicates and bootstrap seed 2024.

The four bootstrap SHA-256 values are:

| Dataset | SHA-256 |
|---|---|
| ASSIST17 | `f0d2c95f7ccbf0d882b160adc9a6d29e9fa56f122073d48bbd66b2e108b952b9` |
| MOOCRadar | `d8358d5a5885267c6818ba10b18f75fcc54f0606ce307ee67e6b78e65c997620` |
| XES3G5M | `80e4c760d2ae45d1eaaebf58247c53519e0427920fe94df20cf7730e80cc8404` |
| Junyi | `0da4a030cba7c5b1f8faa0e8c1d7c7281b252a63659bc47f8e837fcb6851fa2c` |

## Runner command

```bash
bash scripts/run_factorized_requirement_factorial.sh \
  --data-root /home/xph/jwc/research/knofield_data \
  --output-root results/goal_two_module/factorized_requirement_factorial_v10 \
  --stage history_gate \
  --devices cuda:0,cuda:2,cuda:3 \
  --max-parallel 3
```

Formal execution additionally required `--expected-commit <sha> --execute`.
It is intentionally impossible to select `full_factorial` or provide the old
`--gate-approved` switch.
