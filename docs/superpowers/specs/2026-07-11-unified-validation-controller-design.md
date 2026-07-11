# Unified validation controller redesign

## Goal and trust boundary

Replace caller-computable authorization and caller-authored proof files with a
controller-owned, repository-bound execution state machine. The controller
trusts only its fsynced state directory, exact registered Git commit,
controller-copied cohort/manifest/baseline, registered validation data
snapshots, and artifacts created by outer-campaign processes it directly
launches and immediately snapshots. Caller-created decisions, statuses,
summaries, paths, and internally consistent hashes have no authority.

Task 9 is a new registered iteration. Task 8 results remain exploratory.

## Exclusive registration

`controller-init` exclusively creates the state directory and registers:

- a random controller ID and exact current route HEAD;
- canonical copies/hashes of the cohort, candidate manifest/fingerprint, and
  complete ordered baseline rows;
- an explicit validation data root and exact train/valid/Q/holdout-assignment
  paths and single-open fingerprints for every ordered dataset/split;
- an explicit artifact root under which the controller alone allocates
  per-dataset/per-split outer attempts;
- cursor, successes, final zero-delta threshold, issuance counter, and launch
  state.

Registration recursively rejects test fields/paths. State, input copies,
capability registries, launch journals, and proofs use exclusive creation or
write-fsync-rename-directory-fsync under one `flock`.

## Capability lifecycle

`authorize` accepts only controller state, repository root, and its output. It
issues random standard/holdout capabilities only for the next ordered dataset.
Each exact payload is stored in `issued/`; tokens contain no self-authentication
hash. Issuance uses an O_EXCL output reservation plus a controller-owned pending
journal so interruption is idempotently recovered or re-emitted.

`run-split` may consume a capability only while a matching controller-owned
launch journal is in `running` phase. It verifies exact route, counter,
dataset/split/nonce, registered data root, and that `CAMPAIGN_ATTEMPT_DIR` is the
unique new attempt beneath the registered artifact path. It first durably
stages the complete consumption record and then atomically publishes it in
`consumed/`; consumed existence is authoritative if issued cleanup is
interrupted. Fabrication, reuse, stale counters, manual invocation, and route
drift fail before child output/GPU/data work.

## Controller-owned launch and proof

`run-pair` accepts no data, attempt, status, summary, command, or decision path.
Under the controller lock it registers an irreversible launch ID and the exact
two outer commands, then directly starts the repository's existing remote
campaign runner for standard and holdout using the registered inputs and
capabilities. Tests use an actual controlled temporary subprocess/runner and
real attempt file lifecycle, not a mocked completion object.

The controller requires each process to exit zero and create exactly one new
attempt at the registered split artifact root. Any split/outer failure or
controller interruption permanently blocks the iteration; attempts remain as
negative records and no manual status can recover progress.

Immediately after both children succeed, the same controller process freezes
proof. JSON snapshots use one file descriptor: `fstat`, read bytes, second
`fstat`, stability check, then size/hash/JSON parsing from those same bytes.
Dataset snapshots stream/hash one descriptor with the same before/after
stability check. No proof field is hashed and later reopened for parsing.

Each outer status must bind the exact controller-launched command, route, seed,
cohort/manifest, and the exact registered data fingerprints. Its summary output
record must match the same-byte summary snapshot. Summary validation requires
controller/counter/nonce, dataset/split, canonical manifest/fingerprint,
cohort, seed 42, valid-only routing, nonempty mastery shape, finite final loss,
and positive mastery-loss weight. The proof is fsynced before progress advances.

## Gates and stopping

Candidate metrics are compared only with the controller-owned baseline row. A
dataset succeeds only when standard and holdout overall AUC and weighted DOA do
not regress, while exact-zero AUC and ordinary DOA strictly improve. Remaining
is derived from the fixed cursor. If `successes + remaining < 3`, the iteration
blocks. Final global success additionally requires at least one zero-AUC delta
of `0.001` or greater. Only a successful `run-pair` may advance progress.

## Tests and scope

TDD covers exclusive registration; forged/reused/stale capabilities; route
drift; only-next issuance; pending-issuance recovery; atomic complete consume;
manual run-split/proof rejection; controlled-Popen provenance and unique new
attempts; outer failure permanent blocking; single-open TOCTOU injection;
exact manifest cardinality; baseline/test rejection; every independent metric
boundary; and positive/negative final threshold cases.

No Task 9 controller is initialized, no real campaign/training is launched, no
real test data is read, and nothing is pushed by this implementation task.
