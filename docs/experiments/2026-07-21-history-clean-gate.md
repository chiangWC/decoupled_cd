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

## Planning command

```bash
bash scripts/run_factorized_requirement_factorial.sh \
  --data-root /home/xph/jwc/research/knofield_data \
  --output-root results/goal_two_module/factorized_requirement_factorial_v10 \
  --stage history_gate \
  --devices cuda:0,cuda:2,cuda:3 \
  --max-parallel 3
```

Formal execution additionally requires `--expected-commit <sha> --execute`.
It is intentionally impossible to select `full_factorial` or provide the old
`--gate-approved` switch.
