# RCPK standard-only confirmation result

## Decision

RCPK is retained as the first standard-only paper-module candidate. The frozen
architecture obtains three row-aligned external standard-validation wins, and
the real-relation mechanism has a clean, statistically positive effect on two
of those winning datasets plus NIPS34.

The stricter preregistered condition that every counted external win also have
a `0.002` real-versus-control module effect is not yet met: XES3G5M wins the
external comparison, but real relations tie its strongest rewire. RCPK is
therefore retained for a second-component search rather than declared the
finished paper model.

The previous concept-holdout rejection remains unchanged. These results do not
support a TKC/UKC completion or cold-start claim.

## Real-relation attribution gate

Full is compared with Q-only and three relation-type and exact-degree-preserving
rewires. All variants share trainable parameters, data order, masks, optimizer,
dataset recipe and initialization hash.

| Dataset | Full S | Q-only | Best rewire | Delta vs strongest | Delta vs rewire | Brier delta vs strongest | Gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ASSIST09 | 0.795594 | 0.761245 | 0.761030 | +0.034349 | +0.034564 | -0.011796 | pass |
| NIPS34 | 0.784829 | 0.777733 | 0.774884 | +0.007096 | +0.009945 | -0.003051 | pass |

For NIPS34, Q-only has the higher AUC among the four controls, while the table
also reports the real-versus-rewire contrast required to establish relation
semantics. Full exceeds every control on both datasets; the frozen semantics
gate passes.

## Pool expansion

| Dataset | Full S | Strongest complete control | Module delta | External S | External margin | External S win |
|---|---:|---:|---:|---:|---:|:---:|
| ASSIST09 | 0.795594 | 0.761245 | +0.034349 | 0.776432 | +0.019162 | yes |
| NIPS34 | 0.784829 | 0.777733 | +0.007096 | 0.788478 | -0.003649 | no |
| XES3G5M | 0.789947 | 0.790030 | -0.000083 | 0.786396 | +0.003552 | yes |
| Junyi | 0.831026 | 0.828542 | +0.002483 | 0.820434 | +0.010592 | yes |

The table uses the strongest complete neural control for the module delta:
Q-only on ASSIST09/NIPS34 and the strongest rewire on XES3G5M/Junyi. The same
architecture therefore has three external standard wins. XES3G5M is explicitly
an architecture/base win, not evidence for RCPK.

## Paired uncertainty

Two thousand deterministic student-clustered paired-bootstrap replicates were
run on aligned validation rows:

| Dataset | Module delta | 95% CI | P(delta > 0) |
|---|---:|---:|---:|
| ASSIST09 | +0.034564 | [0.029403, 0.039523] | 1.000 |
| Junyi | +0.002483 | [0.001049, 0.003803] | 1.000 |

This is prediction uncertainty under one frozen model seed, not multi-seed
training.

## Integrity

- Model seed is 42; test metrics are null in every new artifact.
- Full/control initialization hashes match within each dataset.
- Target responses are hidden from training context by the frozen 20% context
  target protocol.
- Checkpoint re-evaluation reproduced every summary AUC exactly before
  bootstrap.
- Generated graphs, checkpoints, predictions and bootstrap JSON remain under
  `results/` on xph and are not committed.

## Next use

RCPK remains frozen. The next search may test a separate upstream History Set
Representation component, but may not modify RCPK, add a residual prediction
head, or use a dataset-specific route. If no second component passes, the
standard-only model can still use RCPK as its sole claimed novel module, with
the XES attribution limitation reported explicitly.
