# BETA-CD (NCD backbone) provenance and adaptation notice

The standalone baseline in `models/beta_ncd_baseline.py` and
`scripts/train_beta_ncd.py` adapts BETA-CD with the NCD response function
implemented in the authors' PyAT repository:

- Paper: *BETA-CD: A Bayesian Meta-Learned Cognitive Diagnosis Framework for
  Personalized Learning*
- Paper page: https://ojs.aaai.org/index.php/AAAI/article/view/25629
- Upstream repository: https://github.com/AyiStar/pyat
- Pinned upstream commit:
  `8e19a3759548bebd39f37a34977b1a2148f3c19c`
- Consulted upstream files:
  `pyat/model/base/ncd.py`, `pyat/model/hyper.py`,
  `pyat/model/meta/abml.py`, and `pyat/model/_utils.py`
- Upstream license: MIT

This repository does not vendor the full PyAT package, Hydra configuration,
data loaders, trainer, or evaluator.

The following mechanics are retained from the pinned author implementation:

- The NCD response head uses Q-masked
  `sigmoid(item_difficulty) * (sigmoid(student_state) -
  sigmoid(knowledge_difficulty))`, with no factor-of-ten scaling.
- Each positive linear layer uses the absolute value of its weight, and the two
  hidden layers use ReLU and dropout before the final sigmoid.
- NCD weights use Xavier-normal initialization.
- The shared diagonal-Gaussian mean uses a standard-normal initialization and
  its log standard deviation uses `Uniform(-4, -3)`.
- A local posterior begins as a clone of that shared prior. Following Eq. (5),
  each inner objective averages across Monte Carlo samples after summing the
  response negative log-likelihood across the complete support set, then adds
  diagonal-Gaussian KL exactly once. The three step-specific inner learning
  rates are learnable and initialized to 0.1.

The following are explicit protocol or paper-alignment revisions and must be
disclosed with results:

1. Optimizer-train students receive a deterministic, student-exercise-group
   atomic expected 80/20 support/query split with split seed 2024.
2. Validation students are disjoint from optimizer-train students. Their local
   posterior is initialized from the shared prior and adapted only on their
   support responses; no student-ID parameter is learned or checkpointed.
3. Exactly three inner updates each consume the complete support set. The
   response likelihood is a product over support responses, so its negative
   log-likelihood is a sum, not a mean over the support length.
4. The outer objective consumes only the disjoint query set and implements the
   stable joint negative log mean predictive likelihood from Eq. (3), with no
   division by query length. Higher-order gradients pass through all inner
   updates.
5. The fixed recipe uses four Monte Carlo samples for both inner adaptation and
   query prediction, KL weight `1e-4`, task batch 8, meta learning rate
   `1e-4`, item/head learning rate `1e-3`, and model seed 42.
6. Q rows are unioned per exercise before constructing the binary Q matrix.
7. Evaluation Monte Carlo noise is fixed independently of student identity,
   and checkpoints and row-level predictions are content-hashed.
8. All six manifest files are hash- and row-verified, but this validation-only
   entry point never parses either test interaction file.

## Reduction and update contract

The paper defines the response-set likelihood as a product over responses.
Consequently, the local loss used by this adaptation is

```text
mean_over_MC(sum_over_support(response NLL)) + eta * KL(posterior || prior)
```

with one KL term per complete-support update and three updates in total. The
meta-loss is the negative logarithm of the Monte Carlo mean of the joint query
likelihood; it is not normalized by the number of query responses.

This follows the equations and Algorithms 1--2 rather than reproducing one
implementation detail in the pinned PyAT trainer. The pinned configuration
sets `inner_sgd: true`, and its ABML implementation performs a singleton
update for every support response inside each of the three inner epochs. That
behavior is order-sensitive, applies KL once per singleton update, and is not
equivalent to the paper's three complete-support updates. It may be studied as
an author-code behavior check, but must not be reported as this paper-aligned
adaptation.

The architecture fingerprint records the inner response reduction, KL
application frequency, and outer response reduction. Checkpoints produced by
the earlier support-mean implementation are incompatible performance
artifacts and must not be reused as results for this version.

Accordingly, results must be named **BETA-CD (NCD backbone, paper-aligned
adaptation from the MIT author implementation at the pinned commit)**. BETA-CD
is the framework name and NCD is the selected response-model backbone. Results
must not be described as an unmodified official reproduction.

## Upstream MIT license

MIT License

Copyright (c) 2023 AyiStar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
