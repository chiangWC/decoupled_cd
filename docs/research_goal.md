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
