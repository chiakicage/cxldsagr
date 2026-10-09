# Executable staging integration plan

1. Coordinate the raw bridge ABI with `audit_cold` and the callback contract with
   the parent. Raw h2d accepts local NH entries; no negative host ID is consumed.
2. Add `official_prefetch.py` with strict shape/headroom checks, source-bound
   compilation, preparation-token consumption, stage ownership, registered
   cleanup, raw official call, and promotion. The parent owns the offset EMA and
   context/shape dispatch. The wrapper uses `offset[1:2]` without modifying it.
3. Add `csrc/official_prefetch.cu`: page expansion/reset, exact D2D promotion,
   matching-tag cleanup, and TVM FFI entry points on the current torch stream.
4. Add focused numerical/state tests in the adjacent tests directory. Tests
   cover attempted counts 0, 1, 17, 64 and >64, legal FIFO victims, host zero,
   discontiguous page mapping, and cleanup after no/partial/all publication.
5. Compile and run the GPU3 command from `integration_task.md`. Run the raw
   official integration check once its independent bridge is available. Do not
   use GPU0 or write experiment results from test execution.
6. Return exact workspace and traffic formulas, source/compiled identities,
   callback lifetime requirements, and full-model transition-proof fields to
   the parent. Record results in `integration_checkpoint.md`; explicitly leave
   full-model numerical, cold timing/profile and publication gates to the parent.
