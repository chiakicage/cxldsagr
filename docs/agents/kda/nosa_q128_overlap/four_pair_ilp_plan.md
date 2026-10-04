# Four-pair ILP preparation record

Root selected isolated implementation and preparation on 2026-10-04. GPU execution
remains on hold until selected-source, compiled SASS/resource and independent
review gates pass. The earlier [proposal](independent_warps_ilp_proposal.md) remains
unchanged; its unselected status records the earlier decision boundary.

The new external stage is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_four_pair_ilp_candidate_20261004/`.
Its external execution materials are `task.md`, `draft.md` and
`implementation_plan.md` within that stage.
The source parent and previous diagnostic plans/helpers remain frozen. The new
stage's preparation materials will be bound with its final reviewed manifests;
current compilation does not constitute GPU acceptance.

The candidate changes only halves2 full-stripe scheduling: each lane loads four
K/V `uint4` pairs before its first dependent store, using the original addresses.
Partial and empty stripes retain the original loop and publication chain. The
producer/consumer limits become 80/240 for halves2; halves1 retains 24/240 and must
retain the same selected compiled body. Queue ordering, worker claims, math,
selection, unique payload, masks and numerical repair remain unchanged.

Use original unpadded host allocations and record their actual pointer residues.
The aligned-buffer diagnostic is evidence for the decision, not the allocation
policy for this candidate. Preserve the real `_module` loader and source-derived
build/native identity checks; no production runtime or allocator edits are included.

Preparation gates are a CPU physical-vector bijection/bounds audit over page
lengths 0–64, exact source-scope and partial-loop checks, exact-loader compilation,
selected sm90a SASS with eight host loads before dependent stores, register/stack
inspection and the actual per-CTA register-pool check. Independent review must
confirm those gates before root allocates GPU work. Prepared hardware gates cover
partial/empty stripes, numerical output, exact payload and unique reads, queue,
ownership/lifetime and cleanup, followed by captured exact controls and separate
per-sample page/stripe overlap checks. Every internal sample must meet both 0.90
gates. API latency requires a passed screen and a separate root decision.

The aligned diagnostic removed the host sector excess while its first internal
sample remained below the overlap gate. This motivates testing more independent
host loads; it does not predict an ILP speedup or attribute all waiting to host
loads. The producer budget also affects TMA warp0. No candidate promotion or
serving-performance claim follows from preparation.

Preparation completed on 2026-10-04. The CPU source/address audit covered 3,120
physical stripe cases and 199,680 vector addresses; the partial loop, publication
source and real `_module` loader remain unchanged. Twelve focused CPU fault tests
passed, including host corruption, retained aliases, unknown workspace/device
completion, native argument lifetime, register metadata and the representative
alignment gate. Syntax, Ruff and shell syntax/help checks passed.

The independent selected-image review confirmed all eight halves2 host loads
before the first dependent store, disjoint payload registers R20–R51, the common
partial/full publication path, and producer80/consumer240. Halves1's 5,144 encoded
instructions and the priority compactor remain identical to the actual parent.
The selected halves2 main reports REG240, STACK0 and no LDL/STL; the parent's
STACK32 became zero. Thus the producer-budget and stack effects remain confounded
with ILP. This statement concerns the selected main, not every function in the
library. Dynamic SharedStorage sizes are 197,728 and 181,344 bytes for halves1/2;
actual runtime attributes and cooperative occupancy still require GPU checks.

The representative comparison requires both original CPU and mapped pointers to
have residue16 modulo32; other residues reject before the call, without allocation
retry or substitution. Partial probes record their natural residues. This gate
controls the diagnostic comparison and does not impose a runtime allocator policy.
The helper retains original host owners and all native input tensors/selection
when GPU completion cannot be confirmed.

The final sibling helper is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_four_pair_ilp_screen_20261004/`.
Its `prelaunch_manifest.json` SHA256 is
`667b597d9f967e89e7e5030a1f7876d83d8c963125dfbf8b47b2db1b9564b08a`.
The native stage's `compile/prelaunch_review.json` SHA256 is
`75a2ba50c35b9d0b6eb284530d0b24805327c2f54586d52b5ee8d8cb2b9aea99`.
The CUDA-hidden freeze and both partial/captured-input prerequisite audits exited
zero. Their command receipts and separate logs are under the helper's
`output/nosa_q128_four_pair_ilp_cpu_prelaunch_20261004_01/`.

Root received the reviewed `run_partial.sh` command for exactly twenty GPU4 calls
on CPUs72–79/NUMA1. No GPU call was launched during preparation; GPU correctness,
per-sample internal overlap and complete API latency were unrun at that handoff.

Root subsequently ran `nosa_q128_four_pair_ilp_partial_gpu4_20261004_01`; all twenty
calls and final integrity passed with exit0. Independent raw review checked forty
page envelopes, 104 nonempty stripes, 882 softmax windows and ten exact output-hash
pairs. The lifecycle/native receipt audit rehashed twenty unique native artifacts;
every call preserved both full host-buffer hashes, released four tensor/storage
weakrefs after confirmed completion and restored allocated/reserved HBM to
33,554,432 bytes. Original CPU and mapped host residues were16 in all calls.
Runtime halves2 reports numRegs240, localSizeBytes0, dynamic shared181,344 bytes,
CTA pool61,440 against demand40,960, and one active CTA per SM across132 SMs.
These are GPU4 partial-correctness receipts, not representative overlap or latency.

The read-only lifecycle/native audit is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_four_pair_ilp_receipt_review_20261004/partial_lifecycle_identity_review.json`,
SHA256 `6587530c7aea6a55e641ca85e2d00cfece7c4417b4ca19b1a7d4d0074fae2ae4`.
The independent raw audit is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_four_pair_ilp_review_20261004/partial_raw_review.json`,
SHA256 `e86480dedeedd5e65ca309e365104b227431cbcb3e4e4cbbcff6b7ffc0872721`.
Frozen native/helper materials remain unchanged. The representative GPU0 screen
and complete API latency remain unrun at this checkpoint.

Root then ran `nosa_q128_four_pair_ilp_screen_gpu0_20261004_01`. It exited 2 at
`internal_1`: page and stripe ratios were both 0.7027860967419911, below 0.90.
Both exact controls and all three recorded fixed-FP32 checks passed; each call
preserved the accepted output hash and14,712,832-byte payload. Both original
32MiB host buffers retained CPU/mapped residue16, unchanged full-buffer hashes
and confirmed lifetime cleanup. Independent receipt audit rehashed twenty native
artifacts and confirmed the expected resource/allocator identities. Its artifact
is the receipt-review sibling's `screen_lifecycle_identity_numerical_review.json`,
SHA256 `d48ec4c14abcb9c3e41145743b554e928d82024cfc0d22530e31dbf593c3bc91`.
The candidate is rejected for promotion. No later internal sample, complete-API,
all32 or full32 candidate run followed. This single instrumented sample is not a
statistical latency comparison or an ILP-only causal result. Frozen stage/helper
files remain unchanged; no native candidate is selected this cycle.
