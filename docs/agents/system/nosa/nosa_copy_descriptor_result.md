# NOSA H64K local descriptor reuse result

Completed 2026-10-05 after the active goal's continuation. Reusing metadata
within one guarded input-validation call reduces the ordinary candidate median
by 0.105163/0.106095 ms against the matched captured-copy control at requests
0/16. The whole external implementation reduces it by 0.133753/0.125243 ms
against production. This small effect supports the local hypothesis, but does
not establish a full-serving benefit or meet the remaining MFU/overhead goal.
The prototype remains external; production and accepted experiment results are
unchanged. The next candidate is the [indexer dispatch plan](nosa_indexer_dispatch_plan.md).

The six original formal numerical-reference files needed by the frozen reviewers
are retained separately with their original bytes and hashes. See the
[reference relocation and reopening instructions](../../acceptance/unified_runtime_20261005/t007_reference.md).
Historical commands below remain unchanged; reopening uses only an in-memory
`FORMAL` path override. The relocation does not rerun these diagnostics.

## Comparison and results

B is the unchanged production graph class. C is the captured mode from the
[matched copy control](nosa_guarded_copy_control.md). D changes only that
class's `_validate_inputs`: shape/dtype/device observations are reused within
one invocation; identity, pointer, stride, owner, stream, predecessor and
post-attention checks remain. No observation crosses attention or a subsequent
validation call. C/D retain the same buffers, capture priming and copy policy.

Six independent processes ran B/C/D for request 0, then D/C/B for request 16.
The first triple passed an independent numerical/allocation/source gate before
the reverse triple started; timing did not select continuation. Each process
built an independent empty-cache H65536 prefix in C1024 chunks using the full
32-layer checkpoint, then ran three warmups and seven A128 candidates.
GPU5, CPUs48–55, NUMA1, register8, P65536/NH16777216, precision, GC settings and
the complete `extend_candidate` timing boundary match the preceding diagnostic.
The finite decision, public-output clone, discard, lease drain and allocator
entry/exit checks remain within that boundary. No profiler or delayed submission
was used. All samples, including tails, enter these summaries (milliseconds).

| Request | Arm | Mean | Median | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | B | 19.526233 | 15.335495 | 15.122823 | 37.320087 |
| 0 | C | 15.299718 | 15.306905 | 15.164435 | 15.423078 |
| 0 | D | 15.200745 | 15.201742 | 15.057632 | 15.330021 |
| 16 | B | 19.775127 | 15.507209 | 15.330806 | 37.759029 |
| 16 | C | 15.477014 | 15.488061 | 15.328274 | 15.598954 |
| 16 | D | 15.371986 | 15.381966 | 15.202164 | 15.492610 |

C−B median changes are −0.028590/−0.019148 ms. D−C isolates this concrete local
reuse implementation, including its new dictionaries, under the shared guard
contract; it does not isolate the cost of individual getters. D−B includes the
whole captured-copy/guard implementation. Sequential processes and repeated
identical candidates are not independent workload replications.

Each baseline has two filtered pool scans counted inside candidate brackets;
C/D have none in their short sample loops. These are counters, not scan times.
The mean differences and absent brackets do not establish durable scan or tail
elimination. The unchanged 2,223,619,840 B of diagnostic history clones and the
outside-window saves/checks also differ from ordinary serving. No independent
API time is subtracted from these results or reused as a fresh MFU reference.

## Validation and provenance

All six targets exited successfully. Independent CPU review reopened every
construction output and all 60 warmup/measured outputs: all 66 are bit-exact to
the accepted HBM outputs. It checked all raw sample records, 42 measured samples'
zero device alloc/free/retry/OOM deltas, graph/copy counts, zero HBM host
transfers, recorded native/environment identities and process ordering. Each
target's history bytes/pointers, retained state and synchronized close checks
passed; those history payloads were not saved for independent reopening.

Run IDs are `nosa_descriptor_{baseline,captured_control,descriptor_reuse}_r{0,16}_20261005_02`.
The frozen candidate is
`6d70546ba905e1e7eb0a09e51cd9b5740ccb950353a4102464590f9752facde4`;
the driver is
`1e31edc4c6cb9f883b85c6cdc7d4b6b010bf8aa1358575195322360b2ac2c454`.
All 119 saved source files per run and the final current files were rehashed.
The 118 files from the formal source identity
`93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79`
are unchanged. Only the unused report renderer was added; the complete current
snapshot identity is
`34ce2b2822262d085be6182a8a73887eed267bae203de53ec24c3d0b3754f4fe`.

The initial `_01` baseline exited at this additive-source mismatch before model
loading or timing, with zero samples. Its failed receipt is retained outside
experiments. The corrected `_02` family binds the new full inventory and also
requires exact equality of every original file plus the pinned renderer
addition. No timed sample was retried or replaced.

Raw code, commands, all output tensors and launch receipts are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_copy_descriptor_reuse_20261005_02/`.
The independent analyzer, per-run receipts, all arrays and final source check
are in sibling `nosa_copy_descriptor_results_review_20261005_02/`.
Its `consolidated.json` SHA256 is
`c46d1343ede8b9815f7e28150d5e2aaeae4541072d11c12643e8b7162d4d8283`.
These are engineering diagnostics, not a new four-scheme experiment publication.
