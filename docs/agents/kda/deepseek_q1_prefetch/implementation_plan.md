# Executable plan

1. Add `operators/deepseek_v32/indexer/official_decode.py` and
   `csrc/official_decode.cu`. Include unmodified ECHO headers through their
   original paths; bind source and actual transitive includes to the module
   fingerprint. Expose official scheduler and fused call independently.
   Validate the Python tensor ABI before any native launch. Do not edit
   `echo.py`, shared cache code, or model dispatch.
2. Match the upstream launcher: one SM-count grid, 128 TMA threads, 512 math
   threads, 128 prefetch threads, three Q/KV stages, computed official prefetch
   stages, split KV=256, L2-256B TMA promotion. Use the official metadata kernel
   with batch aligned to 32. Enforce staging size 64, FP32 hint, and pinned
   BF16 host records. Throw on compiler, CUDA, or shape failures.
3. Add `experiments/deepseek_v32_echo_official/src/q1_official_prefetch.py` with
   distinct `check` and `bench` modes. Bind all three real inputs, executable
   sources, installed reference/native dependencies, precision, shapes, GPU
   UUID, CPU affinity, and measurement boundary.
4. Check complete scores and top-k against the fixed mainline paged API and
   the fixed local fused reference when applicable. Check raw staging policy
   independently from any local prefill policy. Test replay with changed Q,
   current keys/scales, host data, and thresholds, after state restoration.
5. After passing checks, measure both native prepared-input invocation and
   the complete standalone wrapper with fresh per-call packing and metadata.
   Both methods restore the same initial state outside event timing; graph
   replay contains exactly one invocation. Report warmup/repetition counts
   and preserve every timing sample.
6. Record findings, supported scope, failures, workspace, and acceptance/run
   identities in `checkpoint.md` and `investigation_log.md`; report the bridge
   to the parent for a separate production integration decision. No baseline
   publication is replaced by this isolated validation.
