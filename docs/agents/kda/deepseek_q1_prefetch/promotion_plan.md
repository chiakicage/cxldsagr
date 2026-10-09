# Executable promotion plan

1. Snapshot baseline CUDA and loader identities into private candidate output;
   inspect actual layer-2 input tensors and preserve their file hash.
2. Implement validate-then-copy/publish candidate as a private CUDA translation
   unit with unchanged exported ABI and prepare/clear kernels.
3. Add `experiments.deepseek_v32_echo_official.src.q1_promotion` with separate
   check, bench and one-call profile modes. Bind source, input and immutable
   binary hashes to a check receipt; bench/profile require that receipt.
4. Run check on GPU1/CPU8-15, including the existing 22 adapter tests using only
   a module-provider substitution plus custom bitwise/alignment/reference cases.
5. Run clean paired graph-event timing only after check passes. Cover P65,600,
   N65,537, count64, zero/mixed/full selected occupancy and consecutive/random
   slots; report samples and median/p90 for the full multi-kernel call.
6. Provide source diff, native identities, command for one-call NCU and results
   to root. Keep production sources unchanged and document verification limits.

Commands share:
`PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 .venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_promotion`.
Use `--help` for explicit paths and mode choices. Independent check output is
under `/tmp/cxldsagr-checks/deepseek_q1_promotion/`; benchmark data lives under
`experiments/deepseek_v32_echo_official/output/data/` with a new run ID.
