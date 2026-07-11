# Unified validation controller redesign

## Goal and trust boundary

Replace caller-computable authorization hashes with a controller-owned,
repository-bound one-time capability registry. The controller trusts its own
state directory, the exact registered Git route commit, a verified cohort and
architecture manifest, a copied baseline table, and outer campaign proofs
tied to capabilities consumed by `run-split`. It does not trust caller-created
decision JSON or an unregistered token.

Task 9 is a new registered iteration. It begins at zero successes with four
ordered cohort datasets remaining; Task 8 results remain exploratory.

## State directory

`controller-init` creates the state directory with an exclusive `mkdir` and
fails if it exists. Under one repository lock it records:

- a random controller ID and exact current route HEAD;
- canonical copies of the verified cohort and candidate architecture manifest,
  plus cohort SHA-256, manifest SHA-256, and architecture fingerprint;
- the cohort's dataset order, cursor, successes, and zero-delta threshold state;
- a canonical copy and hash of complete baseline rows in the same dataset order;
- monotonic issuance counter and the active pair, if any.

State JSON writes use write-fsync-rename-directory-fsync. Cohort, manifest,
baseline, issued capabilities, consumed capabilities, and proofs are created
exclusively or atomically renamed and fsynced. All transitions hold
`controller.lock` with `flock(LOCK_EX)`.

## Capability lifecycle

`authorize` accepts only the controller directory, repository root, and output
path. It never accepts successes, remaining, allowed datasets, decision JSON,
or proof paths.

On the first call it issues capabilities only for the first ordered dataset's
`standard` and `holdout` splits. Each capability has the controller ID, route
commit, counter, dataset, split, and a cryptographically random nonce. The
bearer token contains both capability payloads, while exact copies live in the
controller's `issued/` registry. The token has no self-authentication hash.

`run-split` consumes one capability at its first executable line, before
resolving/creating output or work paths and before GPU or data access. Under
the controller lock it verifies route HEAD, active counter, dataset/split, and
exact payload equality with the issued registry record. It records the
resolved `CAMPAIGN_ATTEMPT_DIR` and expected validation-summary path, then
records the runner's data root and the exact expected train/valid/Q/assignment
paths without opening them, and atomically renames the capability from
`issued/` to `consumed/`. A fabricated,
reused, stale, or route-mismatched capability fails before child side effects.
An outer campaign may already have created a failed attempt shell, but it
cannot read child data, train, or become progress proof.

## Progress proof and gates

A later `authorize` call first advances the previously issued dataset. It
derives both attempt directories and summaries solely from the two consumed
registry records. For each outer `status.json`, it requires completed/exit 0,
seed 42, the exact registered route commit, exact cohort and manifest immutable
input fingerprints, and dataset fingerprints. The registered summary path,
size, and SHA-256 must match the status output record and current file bytes.
The status dataset manifest must contain exactly the paths recorded during
capability consumption, and each current file must match its recorded size and
SHA-256.
Each summary must bind the registered cohort, manifest/fingerprint,
controller ID/counter/capability nonce, dataset, split, seed 42, and
validation-only input role. The controller stores an immutable proof containing status, summary,
data, and output hashes before advancing.

Candidate metrics are compared only with the controller-owned baseline row.
A dataset is a success only when all five conditions hold:

- standard overall AUC does not regress;
- holdout overall AUC does not regress;
- weighted DOA does not regress;
- exact-zero AUC strictly improves;
- ordinary DOA strictly improves.

After proof recording, `remaining` is derived from the fixed cursor. If
`successes + remaining < 3`, authorization fails closed and no new capability
is issued. Otherwise only the next ordered dataset pair is issued. After the
last dataset, final global success additionally requires at least one exact-zero
AUC delta of `0.001` or greater; completion writes state/proof only and issues
no further capability.

## Tests and scope

TDD regressions cover exclusive initialization, only-next-pair issuance,
forged caller-rehashed tokens, one-time reuse, stale capabilities, fake
decision inputs, route mismatch, attempt/output hash mismatch, immutable proof
closure, and consumption before output/GPU/data side effects. Existing cohort
tamper, routing, no-overwrite, and GPU-lock regressions remain.

The redesign changes controller/runner code, focused tests, ledger wording,
and the Task 8 fix report only. It launches no training, reads no real test
data, does not push, and does not start Task 9.
