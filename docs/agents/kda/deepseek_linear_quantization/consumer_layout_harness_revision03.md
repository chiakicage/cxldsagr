# Consumer-layout harness revision 03

Check 02 passed the empty fixture, then stopped in a guarded launch after the
first nonempty official-oracle call. The unchanged quantizer identity check
reported a compiler identity change. That failed process did not save identity
snapshots; the exact change below was reconstructed with the installed setup
functions in a CPU-only process, rather than recovered from the failed process.

Torch Inductor's `_set_triton_ptxas_path()` added `TRITON_PTXAS_PATH` and the
corresponding executable hash to `build_info()`. It selected Torch's CUDA 13.0
ptxas, while the preceding production-style SM90 Triton launch used Triton's
CUDA 12.8 ptxas. The executables have different hashes. The installed libdevice
setup made no change (`use_pytorch_libdevice=False`). The unchanged
`Identity.observe()` rejected the reconstructed delta with the same error.

Revision 03 is prepared at
`/tmp/deepseek_linear_scale_layout_candidate_20261006_03/`. Before either arm's
build identity is captured, the harness resolves Triton's current SM90 ptxas,
records its path/version/hash, and explicitly declares that same path. It then
calls the installed Inductor setup functions and requires zero identity delta
and identical effective compiler bytes. Thus the compiler used by the baseline
remains the production default. Setup evidence is saved before execution and
bound across check, bench and profile. Subsequent identity checks are unchanged.

CPU checks verified that declaring the existing path adds only the environment
field: the same executable was already covered by the scoped dependency hashes.
Inductor setup caused no change before or after identity capture. CUDA remained
uninitialized on cores 32–39 with eight threads; only compiler version queries
were executed, with no kernel compilation. All eight numerical source files,
the fixture generator, and existing comparison/measurement functions remain
byte-identical or AST-identical to revision 02 as applicable. Revision 02 and
its failed run are preserved. A fresh full GPU check remains required.

Evidence is in `identity_diagnosis.json`, `compiler_setup_cpu_checks.json`,
`source_equivalence_revision03.json` and `freeze_manifest.json` in the private
workspace. `ROOT_RUN.md` gives the new observer commands and SSD check output
`deepseek_linear_scale_layout_check_20261006_03`. There is no production
promotion or new numerical/performance result in this revision.
