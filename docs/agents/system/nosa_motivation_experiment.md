# NOSA motivation experiment contract

The new `experiments/nosa_motivation/` runner uses fixed P/NH semantics. Its default
H/A/chunk/users/rounds/P/NH values are 65536/128/1024/16/2/65536/16777216. HBM uses
a P-token whole-session history quota; offload uses NH history pages while sharing
P historical slots per layer. All candidates are GPU transient. No byte budget
adapter can satisfy the fixed-capacity gate.

Backend integration target: `models.nosa.fixed_serving.NosaFixedServingBackend`
with constructor `(model, scheme, chunk_size=..., sparse_pool_tokens=P,
host_arena_tokens=NH)`. Names map `sync_sparse` to `serial_sparse` and `async_sparse`
to `overlap`. `PersistentGRRunner` receives explicit H, A, and H+A limits, native
token validation, and no byte budgets. The resource plan must expose exact quota
fields and `candidate_persistence="gpu_transient"`. Transfer metrics must retain
prefix/candidate distinctions and expose `retained_length`.

Experiment acceptance is independent from goal completion. The four-method run
may be accepted as latency/numerical/capacity evidence only after every request
completes and every offload hidden tensor passes comparison with independently
constructed HBM history. Full model shape, no omitted checkpoint keys, no host
candidate persistence, and exact visit/admission semantics are checked. Its
report explicitly leaves MFU and compute/IO-dominance acceptance pending until
the HBM API harness and complete request profiles prove them.

Raw matrix API MFU must have independently timed matching matrix work covering
the complete numerator. Whole graph/indexer/attention API durations include
normalization, activation, preparation and repair; report their compute-API MFU
separately. Their larger denominator cannot substitute for raw-matrix API MFU
to satisfy the requested proximity gate. The row and report formulas currently
produce only request/candidate wall MFU using the declared dense BF16 peak.

The profile now joins an independent all-layer GEMM/BMM reference with complete
compute API device spans and accepted uninstrumented runner latency. Raw GEMM/BMM
excludes sparse attention QK/PV and keeps its own subset numerator. Raw matrix plus
profiled attention is explicitly a composition estimate, with no automatic MFU
proximity verdict. Scope/activity coverage and raw medians are independently
recomputed. GPU API spans retain intra-invocation gaps but exclude earlier host
work. Raw-only binary additions are separate from replay binary identity.

The runtime and report audit use atol=rtol=0.016, matching the existing full NOSA
checkpoint test, while recording exact equality and absolute error separately.
Every output is saved outside timing. The report rereads tensors and hashes.

The backend is instantiated separately for warmup and measured runs while
retaining the loaded immutable model. It is closed after each lifecycle. Warmup
executes two different users and revisits the first; at default P=H, offload must
observe a real candidate host recall. All measured caches then start empty.

The runner is wired to the fixed backend. Formal execution acceptance still
requires current-source H65536/A128 full-checkpoint validation, finite-cache
eviction coverage, four complete method traces, complete matrix API/MFU evidence,
CPU launch attribution, and internal overlap evidence. Separate engineering
diagnostics do not fill those experiment gates. README reports no formal results.

Measurement schema v2 includes actual native build API metadata, independent
fresh CUTLASS/FlashInfer/TVM-FFI/DLPack/CUDA header hashes, CUTLASS dirty state and
compiler identity. Mapped project, FlashInfer, PyTorch and CUDA matrix/runtime
libraries are read from `/proc/self/maps`, checked against file device/inode and
hashed after warmup and execution. Source, native build/dependency, checkpoint
and binary drift reject the run. This inventory excludes driver-generated device
code; checkpoint shards retain the explicitly weaker size/mtime boundary.

`profile_audit.py` rereads saved replay hidden tensors, selections, validity masks,
pre-call session ownership tags, page envelopes, stripe intervals and timelines.
It recomputes the CPU miss union and causal-work counts, uses the existing native
work validators, and joins per-layer bytes to candidate transfer counters. It
also checks replay plans, shared storage/reservations and trace-buffer sizes.
Measurement reporting independently joins memory samples with per-request rows
and case peaks; allocated, reserved and device-used quantities remain distinct.

Diagnostic validity, every-applicable-layer/sample >=90% overlap for both page
and stripe windows, and end-to-end async speedup are separate fields. A measured
overlap failure is not a failed diagnostic run. Missing overlap coverage remains
pending. Speedup compares total synchronized uninstrumented request latency over
the complete trace and reports first/revisit groups independently. Matrix-API
proximity and compute/IO-dominance claims remain pending independent evidence.

Both reports and profiles are fully prepared in system temporary directories.
Publication copies only complete staged content and rolls back destinations
created by that invocation if any category fails. Existing destinations are never
overwritten. No formal run or performance report was generated by this update.

Implementation validation: 83 CPU tests passed for the complete experiment
suite, including corrupted artifact rejection, strict sample gates, saved-timeline
recomputation, independent raw-matrix audits and publication rollback. Ruff and
profile/audit CLI checks also passed. This is harness validation, not GPU numerical
or performance evidence.
The real-environment CPU provenance smoke also inventoried dependency headers
and rehashed 17 mapped libraries successfully. It executed no GPU kernel.

The affected-experiment scope, commands and planning estimates are maintained in
[the remeasurement queue](nosa_motivation_remeasurement_queue.md). Old valid
artifacts remain until their replacements are measured, accepted and published.
