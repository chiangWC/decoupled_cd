# V2 per-dataset default config (locked 2026-07-07)

Best known baseline per dataset (max of pre-tuning vs r17/r20 tuning). These are
the DEFAULT baselines; the 6 mechanism experiments (Tr-1/2/3, Mo-1/2/3) layer on top.

Shared stack (all datasets): `--v2-hybrid-readout --v2-monotonic-readout
--v2-ukc-propagation --v2-mastery-aux-weight 1.0`.

| dataset | split-training | extra | std AUC | notes |
|---|---|---|---|---|
| assist_09 | full_batch, 300ep, patience5, +cons0.5 | dim128 | 0.7585 | pre-tuning > tuning (minibatch/T1 hurt a09) |
| nips34_clean | minibatch bs128, lr2e-3, 80ep | dim64 | 0.7833 | T1 aggressive-optim beat pre |
| assist_17 | minibatch bs128, lr2e-3, 80ep | dim64 | 0.7982 | T1 |
| moocradar | minibatch bs64, lr1e-3, 40ep | dim64 | 0.9282 | clean minibatch base |
| xes3g5m | minibatch bs64, lr1e-3, 40ep | dim64 | 0.7928 | clean minibatch base |

cons (ukc-consistency) is full_batch-only, so it is present only for assist_09;
the minibatch datasets use hybrid+mono+m1+aux.
