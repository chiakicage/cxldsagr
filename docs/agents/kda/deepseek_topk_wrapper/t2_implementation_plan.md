# T2 execution plan

- [x] Select official value sort plus conditional exact finite-tie repair.
- [x] Establish uint32 packing guard and complete C5 fallback outside it.
- [x] Receive and hash real selected-value/ID samples and their source metadata.
- [x] Measure finite equal-bit run distributions on CPU; distinguish baseline
  sorted IDs from the unobserved T2 selector emission order.
- [x] Prove packed run/index ordering, zero distinction, invalid masking,
  width/padding and fallback boundaries on CPU, with CUDA uninitialized.
- [x] Write temporary candidate/driver and compile explicit SM90 targets offline.
  Inspect 32-bit compare instructions, registers, shared/local memory and source
  hashes before scheduling the screen.
- [x] In root's granted GPU window, compare all value bits and indices, including
  signed-zero groups, adversarial in-run permutations, long finite ties,
  causal/all-invalid padding, K>N, K=1, noncontiguous input and fallback geometry.
  Check nondefault stream, changed-input graph replay and retained outputs.
- [x] If correct, benchmark complete APIs in alternating order on Q/N=
  1024/1024, 1024/32768, 1024/65536, 128/65664, with causal/tied/invalid cases.
  Use ten warmups and forty repeats, record API wall and event interval separately,
  and save actual loaded vendor/Triton/compiler identities after timed work.
- [x] Reject using new evidence; no production change or report
  replacement before parent-controlled affected correctness/performance gates.

Sample analysis, CPU proof, wrapper ABI checks and the bounded GPU screen
are complete. See [the checkpoint](t2_checkpoint.md) for results and the
executed GPU commands. All prototype files and engineering outputs are under
`/tmp/deepseek_topk_order_t2_20261004/`; production and published reports stay
unchanged. The exclusive T2 GPU window has been released.
