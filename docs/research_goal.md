# Active Research Goal

All model development, commits and experiments run in the xph workspace. The
model seed is fixed to 42 and the existing dynamic dataset protocol and
row-aligned external registry are authoritative. Multi-seed experiments are
out of scope.

The terminal objective is one architecture topology that simultaneously:

- obtains at least three ordinary external wins;
- is a coherent framework whose functional components have explicit
  responsibilities and interfaces.

The number of claimed contribution modules is not fixed. Standard encoders,
diagnosis heads, and other necessary framework boxes may remain non-contribution
components. Every component presented as a contribution must pass clean
ablation on datasets won by the Full model; components with weak ablations are
kept only as implementation details. The final paper claims all and only the
components supported by fair, material ablation effects.

An ordinary win requires all three raw-AUC conditions:

- `S >= external - 0.002`;
- `H >= external - 0.002`;
- `T > external`.

For each claimed module, compare Full with the stronger reasonable direct or
capacity-matched control. Before implementation, declare whether the component
is responsible for overall diagnosis quality, the target slice, calibration,
or another metric directly needed by the external-win objective. The module
must improve that declared primary metric by at least 0.005 on at least two
Full-winning datasets, improve it by at least 0.01 on at least one of
them, and have a student-clustered paired-bootstrap 95% confidence interval
whose lower bound is above zero on at least one Full-winning dataset. No other
S/H/T axis on a Full-winning dataset may regress materially.

The base architecture, module source, number of exploratory candidates and
final paper narrative are not frozen. TKC/UKC is an entry hypothesis rather
than a mandatory story. Literature-derived, cross-domain and independently
designed modules are all eligible. A mechanism is retained only when the Full
model wins externally and its fair ablation is strong; provenance alone does
not qualify or disqualify it. A signal route is not rejected merely because
its input is absent from part of the current pool: if the route has predictive
value and clean ablations on a coherent set of at least three qualified
datasets, the dynamic pool and narrative may be updated accordingly.

Search is validation-driven. Test confirmation occurs only after architecture,
training recipe and ablations are frozen. A short rolling plan may organize the
current candidate, but it must not replace these terminal conditions.

## Standard-only amendment (2026-07-21)

The concept-holdout RCPK screen failed and remains a negative result. Following
that evidence, the active RCPK paper route narrows its declared responsibility
to standard overall diagnosis and no longer uses H/T as a success condition or
as a TKC/UKC-completion claim. This is a disclosed research-question pivot, not
an alternate slice selected to hide a failed holdout experiment.

For this route, the terminal performance condition is at least three wins over
the frozen external standard-test AUC lines from the registered same-split
benchmark runs under one architecture fingerprint. Row-level external
predictions remain mandatory for validation-driven selection and should also be
retained for test whenever available.
The existing module effect thresholds above apply to standard AUC: at least two
externally won datasets improve by `0.005`, at least one improves by `0.01`, at
least one student-clustered paired-bootstrap confidence interval has a positive
lower bound, and any regression on another standard-winning dataset is below
`0.001`.

RCPK meets this amended terminal condition on the frozen 2026-07-21 test
confirmation. ASSIST09, NIPS34, and Junyi are external-plus-module wins;
XES3G5M is an additional external architecture win with a negligible negative
module delta and is not used as attribution evidence. The next phase is paper
validation and comparison, not compulsory module accumulation. The rejected
H/T results remain reportable limitations.

## RCPK mechanism clarification (2026-07-23)

A same-input, same-capacity control confirms that RCPK's target-conditioned
relation gather materially outperforms a student-global relation summary on
ASSIST09 and NIPS34. This strengthens the structural attribution of the existing
standard-only module. The accompanying natural relevance-share analysis rejects
low-share target-irrelevant relational dilution as a cross-dataset problem
statement: ASSIST09 has the opposite direction, Junyi is inconclusive and
NIPS34 has no share variation. Retain the mechanism result, but do not present
relational dilution, TKC/UKC incompleteness, difficulty calibration or efficiency
as an established premise. A paper claim still requires a focused novelty audit
against existing interaction-aware graph CD before the new control is evaluated
on frozen test rows.
