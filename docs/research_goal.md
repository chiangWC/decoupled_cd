# Active Research Goal

All model development, commits and experiments run in the xph workspace. The
model seed is fixed to 42 and the existing dynamic dataset protocol and
row-aligned external registry are authoritative. Multi-seed experiments are
out of scope.

The terminal objective is one architecture topology that simultaneously:

- obtains at least three ordinary external wins;
- contains at least two independent framework modules that pass clean
  ablation on datasets won by the Full model.

An ordinary win requires all three raw-AUC conditions:

- `S >= external - 0.002`;
- `H >= external - 0.002`;
- `T > external`.

For each claimed module, compare Full with the stronger reasonable direct or
capacity-matched control. The module must improve T by at least 0.005 on at
least two Full-winning datasets, improve T by at least 0.01 on at least one of
them, and have a student-clustered paired-bootstrap 95% confidence interval
whose lower bound is above zero on at least one Full-winning dataset. No other
S/H/T axis on a Full-winning dataset may regress materially.

The base architecture, module source, number of exploratory candidates and
final paper narrative are not frozen. TKC/UKC is an entry hypothesis rather
than a mandatory story. Literature-derived, cross-domain and independently
designed modules are all eligible. A mechanism is retained only when the Full
model wins externally and its fair ablation is strong; provenance alone does
not qualify or disqualify it.

Search is validation-driven. Test confirmation occurs only after architecture,
training recipe and ablations are frozen. A short rolling plan may organize the
current candidate, but it must not replace these terminal conditions.
