# Response-Conditioned Path Kernel result

## Decision

RCPK is rejected as a paper module. After repairing the training-history
boundary, real relations produced large standard-validation gains but did not
retain a material gain on concept holdout. Neither first-screen dataset reached
the preregistered cross-split effect threshold and Full added no external win.

No relation rewires, target-slice/bootstrap analysis, pool expansion, or test
confirmation was run because a stronger control cannot reverse the failed
Full-versus-Direct bound.

## Invalid precursor and protocol repair

The first execution used legacy train-edge reconstruction
(`context_target_frac=0`). A supervised response could return to its own item
through a Q/metadata cycle, causing response leakage during training. Full
reached only `0.684102` ASSIST09 standard validation AUC while Direct reached
`0.768095`, despite Full training loss continuing to fall. This output is
marked `invalid_protocol` and is not evidence against the mechanism.

Commit `18fcae13` requires the same `context_target_frac=0.20` for Full and all
controls. The value is inherited from the frozen train-only relation-signal
screen. Hidden training targets are removed from the response mask and concept
evidence before prediction. No architecture or optimizer parameter changed.

## Valid Full-versus-Direct screen

All rows below are validation AUC. Standard and holdout use the same dataset
recipe, seed 42 and 20% context-target masking.

| Dataset | Axis | Full | Direct | Delta | Full external margin |
|---|---|---:|---:|---:|---:|
| ASSIST09 | S | 0.795594 | 0.761245 | +0.034349 | +0.019162 |
| ASSIST09 | H | 0.754286 | 0.752220 | +0.002067 | -0.014661 |
| NIPS34 | S | 0.784829 | 0.777733 | +0.007096 | -0.003649 |
| NIPS34 | H | 0.776733 | 0.776656 | +0.000077 | -0.007516 |

The corresponding Brier deltas (Full minus Direct) are `-0.011796`,
`-0.000510`, `-0.003051`, and `-0.000106`; calibration therefore improves on
all four runs even where AUC gains collapse.

The preregistered primary effect is
`min(Delta S, Delta H)`: ASSIST09 obtains `+0.002067` and NIPS34 obtains
`+0.000077`. The gate required at least one dataset at `+0.005`, while the
other could not regress by more than `0.001`. The first condition fails on
both datasets. ASSIST09 remains below the holdout external line, and NIPS34
remains below both standard and holdout lines, so Full adds zero ordinary
external wins.

## Attribution integrity

- Full and Direct share architecture fingerprint `f0151739cd1cbfaa`.
- ASSIST09 initialization hash is
  `27bfa85029d4b012bca57b78a90928431a235f8d02889ff5d51b6091ace023d0`
  for both variants.
- NIPS34 initialization hash is
  `49313bbb287d2d2e6422db5b8bd46c7d9f18c06379c2b9725e1df0605f43d889`
  for both variants.
- Graph transitions are non-persistent, non-parameter buffers; the trainable
  state and parameter count are identical.
- `test_metrics` is null in every valid artifact.

Valid artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| A09 S Full | `9f38019922033d63057f7a3303865b160bb6a12b4ad7f7b8611299a96c0e797e` |
| A09 S Direct | `b5851b0d70213cb07671835141730141a71c8c4649d7d768b18d5c68d64a015a` |
| A09 H Full | `ab78184b86df7029dce996c0d683204bcffeab163aa68704170d58ad84aed81d` |
| A09 H Direct | `254db1e2bf94f29d71cf7e31df7e0791c646b96c005c580e2a8244fc78cd4457` |
| NIPS S Full | `e0a1db36e4c44c4ebf4557abb6923bd9a689bbd70faf295e2536d56ac9403df5` |
| NIPS S Direct | `5100db134f8ab7114f71416e0830189cb5fdcd9659f0b69c0a7f200936cd108a` |
| NIPS H Full | `906c1688899d4a4ca4bdad2e6e53d5236f2a40777c41f7da63be2d3aae7cede7` |
| NIPS H Direct | `43b115a6f2a6432700195a3fb47237a78c1209eefd791697977ec678b6b1c4ee` |

## Interpretation

The fixed path statistic validates the earlier observation that real
curriculum relations can strongly improve in-distribution representation. It
does not solve the concept-holdout responsibility: the benefit is largely lost
when the interaction split changes. RCPK must not be renamed, augmented with a
gate or residual head, or reported as a qualified component. The next search
should target holdout-stable state inference rather than static overall
representation.
