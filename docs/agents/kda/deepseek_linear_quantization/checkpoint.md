# Linear quantization checkpoint

The user changed the preferred implementation to handwritten Triton on
2026-10-04. Native V1–V3 remain component evidence and none was promoted; V4
stopped after offline work. The current direct Triton candidate follows
[triton_implementation_plan.md](triton_implementation_plan.md) and
[triton_candidate_plan.md](triton_candidate_plan.md). T1 passed both independent
timing windows and was selected for production integration. The production API,
consumer and actual-checkpoint MLP GPU gates now pass. Full-serving latency and
MFU measurements under the new source remain the root task's next gate.

The CUDA C++ `warp1_cta4` prototype passed numerical acceptance but was rejected
after complete-API timing. The immutable C8 reference and latest accepted formal
result C7_hint use the compiled helper; no C8 formal result exists. The pending
production source must not inherit either version's performance claims.
Engineering files for this parent candidate are under
`/tmp/deepseek_linear_native_quantization_20261004/`.

## Offline preparation

`prepare_01.json` passed with 115 deterministic fixtures, no CUDA context before
or after compilation, and both a TVM FFI shared library and standalone SM90 PTX.
Its SHA-256 is
`829b671c0ffb5fa5cfb41831e485303d12cc08be7a545e40c2d232d2cfa4bc6e`.
The compiler-confirmed manifest contains 382 dependencies. Independent review
found no local, CUDA or TVM FFI header missing from the textual JIT key;
unlisted transitive dependencies are system C/C++ headers. The actual full
dependency hashes and build.ninja commands remain in the prepare record.
All three dtype kernels emit the exact reciprocal-448 constant, `mul.rn.f32`,
`div.full.f32` and `cvt.rn.satfinite.e4m3x2.f32`, with no FTZ.

Native source identities:

- `native_quantization.py`:
  `59a28171c3633cce06dee3e636f3ad0348dfdb1b57091d2c410ed0adb3adec6e`.
- `csrc/activation_quantization.cu`:
  `cc48e83b961c4ed03e858d48625eea41500fe85d0e142aaf14519ebc49e6c028`.
- `adversarial_fixtures.py`:
  `3ad364c7fc2161b3e8c46c36bc2aabe7be5b3ee3eae7d87de8af97b7e7973b20`.
- Native binary:
  `f05bb4435dad6196fbfcda5c1f437ce2428dfbf93367863914ccd2370fe3c146`.

## Exactness screen

After independent source review, root granted correctness-only physical GPU 2.
`screen_01` ran in exec session 13074, PID 1904498, and exited 0. The GPU grant
was released immediately after exit. Screen driver SHA-256 is
`6200611e55c52c97a0772382e49829abb4eaf39f6a346811b556c152540b2ed1`;
the result `screen_01.json` has SHA-256
`553afa3484e7a0f27dfff8ac28dbfffd12a9fa47a20cc7aac7390b3c15f26442`.

All 115 fixture manifests matched the prepared matrix and passed exact data
and scale bytes, full input-owner preservation, output ABI, output guards and
retained native outputs. The matrix includes all BF16/FP16 encodings, controlled
group maxima, FP32 scale/FP8 rounding boundaries, nonfinite/signed-zero values,
partial K, positive/zero strides, overlapping read-only views and empty rows.
Six lifecycle cases cover BF16/FP16/FP32 crossed with K128/129: nondefault stream
predecessor/consumer ordering, 12 independent native/oracle captures and 18 paired
changed-input replays, stable borrowed outputs and retained owned clones.

The screen verifies the actual imported DeepGEMM helper's file hash against
the pinned source, records callable/module/compiler identities, and uses fresh
per-run oracle cache roots. Each fixture records generated/changed artifacts
and the available current-run artifact set; 464 artifacts are saved overall.
The accepted native source/dependencies/binary are explicitly bound to prepare,
while the provenance-only screen-driver revision is recorded separately.
Source-before and source-after records are identical.

Root's `independent_screen_audit_root.json` independently passed all required
counts, prepare linkage and 862 current source/binary/PTX/artifact hashes.
This is a record/source audit, not a second GPU run. Root granted numerical
acceptance for the initial candidate. No latency or MFU claim follows from it.

## Measurement and production boundary

The separate `benchmark.py` compares native allocation/launch against the
immutable C8 public `quantize_fp8_activation`, including its validation and
device-context wrapper. It includes 16 aligned BF16 target shapes, separately
paired wall/event observations, complete eager and graph copy/replay/owned-output
APIs, and exact pre/post checks. Borrowed replay is separately labeled as a
diagnostic that excludes input copy and output allocation. `bench_01` completed
under root's exclusive physical GPU 2 grant, exec session 46121, PID 1912666,
exit 0; the grant is closed. All 16 cases, 4800 paired observations and 9600 API
samples passed exactness and source checks. The result SHA-256 is
`e3443f85be79021c53c34851fb606b7c171059a0eefff80dc86cef2d14725f4a`.

The scalar candidate loses all 50 owned-graph event pairs at M1024 and each of
K7168/16384/18432. At K18432 its medians are 84.016 us versus 54.592 us owned,
and 47.552 us versus 19.008 us for borrowed replay. Eager host overhead can hide
this kernel regression, so the candidate is not promotable. The independent
raw audit `independent_bench_raw_audit_graph_profile_01.json` passed every pair,
all 96 summary cells, ordering and source/binary/artifact checks. Its SHA-256 is
`a2a1a06152cf56bdac26ac615441cc184186d7776e8d7144ff9ed485a12d2672`.
The next candidate follows [aligned_candidate_plan.md](aligned_candidate_plan.md).

The single-thread native prototype temporarily sets a process-wide architecture
environment variable during FFI compilation. A subprocess loader was prepared
for possible native integration, but the user's switch to Triton supersedes
that implementation choice. Neither native loader nor native quantizer was
integrated. Contiguous FP32 scales remain unchanged; scale-layout optimization
is a separate candidate.

## Aligned vector candidate

V2 uses `/tmp/deepseek_linear_native_quantization_v2_20261004/`. The accepted
generic kernel and launcher remain byte-for-byte identical to V1; BF16 primary
widths with canonical contiguous strides and valid pointer alignment use vector
loads/packed stores and four groups per warp. The exact dispatch and parent
relationship are in [aligned_candidate_plan.md](aligned_candidate_plan.md).

Offline `prepare_01.json` passed 213 deterministic fixtures and compiler
dependency checks with CUDA uninitialized. Its SHA-256 is
`033005d9c0ab1dff2f1f230854f93ab147864847c4a94a34638e55d5b4f359f5`.
The fast kernel uses 21 registers, zero shared/local storage and no stack;
PTX contains four vector loads and packed stores, eight SATFINITE pair
conversions, and no integer division or FTZ. Independent review checked all
382 actual dependencies, packed byte order, dispatch and unaligned-output fallback.

V2 source identities are:

- CUDA: `3fe26a85ff6442c00720c2654d96d2ea7cf6b84bb33b09f1cf081be463d570f4`.
- Wrapper: `2521c772e1d00bf0f4b2f8fdf4119334f659f2dde5061e0dddb66ef3a8e26554`.
- Fixtures: `f7ccd5f25adeb572e7500e5101a09276244a83897409610e2110ba7ea74ef966`.
- Screen driver: `8dae893d0e85b48a6d74f795b7e3456d9678e89e49a8d2790706467c5141ac4e`.

Root granted correctness-only physical GPU 2. `screen_01` exited 0 in exec
30397, PID 1923023; the grant is closed. Result SHA-256 is
`51d2a2efba8f78b89911c7c82130501d7eb5deaffd4466c70d1e0a8ddeab589b`.
All 213 fixtures passed, including 63 confirmed fast-path cases and their
unaligned-output fallback guards. Eight lifecycle cases include fast BF16
K7168/K18432, with 16 graphs and 24 paired changed-input replays. All exactness,
dispatch, input-owner/output-guard and retained-output assertions passed;
source records are stable and 516 oracle artifacts are identified.
Root's independent record audit rehashed 912 paths and passed every gate.

The benchmark source is
`7d1ee7cb1e3ab54b214dbcb294f70c4054bef7a8d46881e9f6b8e2ab4644d9ca`.
Independent review approved the unchanged paired methodology and the expanded
213/eight-case acceptance gate. `bench_01` completed under a separate exclusive
GPU 2 grant with 10 warmups and 50 pairs (exec 96009, PID 1926097, exit 0); the
grant is closed. Its SHA-256 is
`7778aee9104ac33889ac69efb1b92c50abdcd57330e194d9e884d3fbb3f14b6f`.
All 16 cases and 4800 pairs passed validation and source checks. At
M1024/K18432, owned-Graph medians remain slower than the original public API:
65.776 versus 56.000 us; borrowed replay is 29.072 versus 20.544 us. V2 improves
the parent but is not promoted. The independent raw audit has SHA-256
`5b1e0dff14f69c556b3cff859da8ed3eea76ec578793b50343d1d3543dbf45e0`.

V3 follows [redux_candidate_plan.md](redux_candidate_plan.md). It changes only
the aligned reduction; generic code, packing and geometry are unchanged. The
independent CPU proof checked all 65536 BF16 patterns, 983040 anchored pairs
and 1242408 full groups from the 63 fast fixtures, with identical scale bits.
Proof result: `/tmp/deepseek_linear_redux_review_20261004/scale_equivalence_01.json`,
SHA-256 `809e816e8337d6eeefb821e5abd3fff537264f632d31fa373c9d36ec711ce70f`.
The proof supplements, and does not replace, the new GPU screen.

V3's direct official-helper GPU screen passed all 213 fixtures and eight
lifecycle cases (exec 26745, PID 1932725, exit 0). Its result SHA-256 is
`18d502190bee89e9655c086115601562e6aa578f5cb311b98514b06c0f39cab0`.
Root rehashed 912 evidence paths; the independent review has SHA-256
`0d995f9e29716a3bea1069ffe4141585e67a9571a688693fe56c03e29d9db453`.
The frozen wrapper, binary, driver, fixtures, actual compiler dependencies and
official compiler artifacts form the transitive oracle for the Triton screen.

V3 complete-API timing passed all 16 cases and 4800 pairs (exec 28361,
PID 1936920, exit 0); result SHA-256 is
`332181e3ce49bd98de3440e8763e1f16e2809eaed82fcffea8e83f60aa08ab82`.
M1024/K16384 and K18432 owned-Graph paired ratios remained 0.947 and 0.929.
K18432 borrowed replay was 22.736 us versus 20.592 us. Eager host-overhead
improvements do not waive that Graph regression, so V3 was not promoted.

V4 proposed exact power-of-two reciprocal construction, but its GPU witness
and correctness/performance runs never occurred before the Triton pivot.
The prepared witness record is
`/tmp/deepseek_linear_native_quantization_v4_20261004/witness_prepared.json`,
SHA-256 `724a319d5a25818cec4bbc61373d4407f536e4142976f6f01af5076a526db936`.
T1 deliberately retains `div.full.f32`; offline V4 preparation is not evidence
for replacing the reciprocal in production.

## Direct Triton T1 preparation

T1 lives under `/tmp/deepseek_linear_triton_quantization_v1_20261004/`.
`prepare_02.json` passed with all 213 unchanged fixture manifests, stable
source records, 134 scoped source/compiler identity files, and CUDA
uninitialized. Its SHA-256 is
`747a74ba5a991382ad05b16b6b74dbfb2f72e3587db107e206c7347a46b1a2e2`.
The earlier prepare record predates the final identity-helper addition and is
not accepted for the frozen candidate.

Offline explicit-target compilation covers primary BF16, strided FP32 tails
and a single-element FP16 case. With the 16-byte pointer alignment inferred
for normal allocations, the primary layout is `sizePerThread=[1,16]`,
`threadsPerWarp=[4,8]`, `warpsPerCTA=[2,1]`; metadata requests 128 bytes of
dynamic shared memory. The primary PTX has eight 128-bit input loads, four
128-bit output stores, four `div.full.f32` instructions and SATFINITE casts.
PTX SHA-256 is
`6ebb305a6574222e04c2e5802884cf552163834ec7274fe90105cbe111fc1d68`.
`cuobjdump` reports 51 registers and no stack/local storage. These are offline
compiler observations, not latency measurements.

Frozen T1 source identities are:

- Kernel: `8b666f5832a47f930134067a33cc31325313f08f89d3ef3d3a53671fc62c9768`.
- Wrapper: `311198a1002456de7ce23a21b4dbaa0459120e4b204ecc247bde0c71c30a7584`.
- Identity helper: `5ca4a259d7130de2ebc9d343ec3dd7ceddac94a974216d46aceca2900759aad6`.
- Driver: `14c8a36578b293d7c7d2c196e2d004932dbed6a290a40796ef13d0d2a36599eb`.
- Benchmark: `5b1aab476f17a808b6545f30ee5c8b005878154455ea680f3a34e247984f03ec`.

Independent CPU review passed two identity-helper tests and verified unchanged
lifecycle source SHA-256
`33182b303adf9ef3468a2dc1c8d86a9d1bf9fd3697184cf35d4a076cd0420e5b`.
The benchmark review confirms that capture, timing, paired ordering and all
three measurement boundaries retain the accepted V3 methodology. Review
SHA-256 is `0c5f563413674a37e1a77c64289af950eb68877ea384b6049e628322ff589357`.
T1 `screen_02.json` passed all 213 fixtures and eight lifecycle cases in
exec 13800, PID 1965022, exit 0. Its SHA-256 is
`6162336222f716d9c2bd42dd248161f6a0026ff1c0eb43bf129124c3ccdb19dd`.
The CUDA device name is NVIDIA H200 while `nvidia-smi` identifies the device as
NVIDIA M403; retain both reported identities rather than relabeling earlier
runs. The screen completed 16 independent
Graph captures and 24 paired changed-input replays, with 300 actual observed
Triton specializations and unchanged source identities. Exact data/scales,
input owners, aligned and deliberately unaligned output guards, retained
outputs and stream ordering all passed. Exactness is transitive through the
frozen accepted V3 official-helper screen, not a fresh official compilation.

The first screen attempt stopped before a real comparison because the parent
native loader could not find `ninja`; adding `.venv/bin` to PATH fixed that
environment issue without changing source. `screen_02` uses the same accepted
prepare and source files. The correctness GPU grant is released. Independent
record review passed all 213 manifests, eight lifecycle cases and 300 live
specializations, including metadata, ASM, retained CUBIN and loaded-handle
checks. Review SHA-256 is
`fcefc9043d53ba30fcd44fb783fb03ea92025fef29ef2d818ea009422a7f5c97`.
The current benchmark admission gate also passes. No complete T1 performance
measurement is accepted yet.

The first paired timing attempt stopped after one shape when the identity
guard detected TorchInductor setting `TRITON_PTXAS_PATH` to its bundled PTXAS.
No partial timing is accepted. CPU reproduction found no libdevice change;
Torch PTXAS SHA-256 is
`daba837a68265cae38c832d13399b61dab811891de9b8914defddef143b849f2`, while
Triton's bundled PTXAS is
`c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`.
The differing compiler identities were not normalized away.

Root selected the Triton bundled compiler, explicitly configured before either
API starts. This is a controlled comparison of both APIs with the same compiler;
it is not a rerun of prior C8 default-compiler timings. `prepare_03.json` passed
with unchanged candidate/harness source and SHA-256
`f1cca195ce7f25640fe091dc571b9e3fe91efb63421a51bf41336fb3d508bfe9`.
The fresh `screen_03.json` passed all 213 fixtures and eight lifecycle cases
under that explicit compiler identity (exec 52861, PID 1971596, exit 0).
Its SHA-256 is
`ccbd8b282bdcdc415e3ee90ee8fefac2627ffde8ad65291d97abfeb743638ea3`.
The correctness GPU grant is released. Independent review passed with
SHA-256 `29d79b6a78e7bf788a87afb094bb7f4904fdb32171abd57f40809b5e428ecc09`.

Under a separate exclusive GPU 2 grant, `bench_02.json` completed all 16 shapes
and 4800 pairs with 10 warmups, 50 repeats and order seed 20261004 (exec 25336,
PID 1975640, exit 0). Result SHA-256 is
`3a6c4ea29417817af5f83a39e246a27957023ab4573b7f03503e7676198165d6`.
All numerical, lifetime and source checks passed. Eager-wall paired median
speedups span 1.489–1.958; owned-Graph event ratios span 0.990–1.032.
M1024/K18432 owned medians are 55.408 us for T1 and 57.008 us for the original
public API, with paired ratio 1.027; borrowed replay has paired ratio 1.089.
The chosen shared PTXAS boundary applies to every comparison. The timing grant
is released. Independent raw audit passed with SHA-256
`c876265843eefe1b51accaca700c7a50e071b540e7e4bbe2132f953e7f3eca09`.

Confirmation `bench_03.json` uses order seed 20261005 with the same 16 inputs,
10 warmups and 50 repeats. It passed all 4800 pairs in exec 42076,
PID 1977441, exit 0; result SHA-256 is
`e572a7d1060790b4b1ff276a46a0248b0179e70d95b8556c85cfe7e8048fd46c`.
Eager-wall paired ratios span 1.471–1.969, owned-Graph event ratios
0.986–1.030, and borrowed replay ratios 1.000–1.089. M1024/K18432 owned
medians are 55.504 versus 57.168 us; borrowed medians are 19.392 versus
21.056 us. Independent raw review passed with SHA-256
`fe87a5a18b0102388a4de7f390da1a8fc233a41b5921ee2f71917695df1acc20`.
Root selected T1 because eager improvement repeats, complete Graph API
performance is preserved and primary large shapes improve. These component
values do not establish a full-model latency or MFU improvement.

The production integration copies the unchanged kernel to
`operators/deepseek_v32/linear/_quantization_kernel.py`, exposes the CPU-safe
wrapper in `quantization.py`, and keeps source/runtime identity in
`_quantization_identity.py`. The public `fp8.quantize_fp8_activation` delegates
the GPU contract once while preserving CPU validation and reference arithmetic.
AST/source comparison confirms that every other function/class in `fp8.py`
matches frozen C8, apart from removal of `_compiled_quantizer`; the kernel
bytes remain identical to the selected T1 source. No SFA or projection sharing
is included.

## Accepted production integration

The production source identities are:

- `fp8.py`: `a3f2e5c0ea30a93cb253124d8fe787c973b091b31be01840fd143bb04ec2efe6`.
- `quantization.py`: `cda887ae747dd2958d98eb15ca52ff0db72544bc50332953247d75a4d5c9c2b4`.
- `_quantization_kernel.py`: `8b666f5832a47f930134067a33cc31325313f08f89d3ef3d3a53671fc62c9768`.
- `_quantization_identity.py`: `d1ae02034fd50ff8757172c8983fa83e010f0183ed861178e64fa9d40e810696`.

The helper adds installed non-Triton-prefixed compiler knobs to its stable
environment record: `CC`, `DISABLE_PTXAS_OPT`,
`LLVM_EXTRACT_DI_LOCAL_VARIABLES`, `NVPTX_ENABLE_DUMP`, `PTXAS_OPTIONS` and
`USE_IR_LOC`. It retains the original loaded-source identity, actual returned
CompiledKernel objects, loaded handles and retained CUBIN hashes. Three
permanent CPU identity tests pass; production source passes Ruff. The focused
CPU run passed nine tests with 28 GPU cases deselected. The combined public
and consumer CPU run passed 30 tests with 41 GPU cases skipped, which is only
CPU evidence.

The approved public regression suite was copied into the linear tests and
then renamed byte-for-byte from `test_quantization.py` to
`test_linear_quantization.py` after its source-bound GPU run. This avoids a
pytest module-name collision with the indexer suite. The old autouse compiler
reset fixtures were removed from linear and packed MLP tests; resets scoped
to the independently compiled official test oracle remain. Root owns the
separate 34-case public GPU suite, small C8 checkpoint comparison and combined
H64K acceptance records.
The renamed public test retains SHA-256
`cb8a7b9c7b31e9e661b52f1e7ab9ed23567bba402b6015a029badbb25de8df29`.

Production consumer evidence is under
`/tmp/deepseek_linear_triton_production_20261004/`. Its `run.py` reuses the
unchanged numerical/lifetime assertions in
`/tmp/deepseek_mlp_packing_integration_20261004/check_checkpoint.py` and binds
the new quantizer source/compiler/runtime identities. The accepted runner
SHA-256 is `3f10f1396439984d8269556ce5131edb5fdabe80e64edefef04f76458104b778`.
Both GPU 3 runs declare these paths before Python starts:

```text
TRITON_PTXAS_PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
```

The first path selects the actual SM90 compiler, with SHA-256
`c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`.
The second predeclares FlashInfer's auxiliary Blackwell setting so lazy import
does not mutate the recorded process environment; its binary SHA-256 is
`a0212ca1a74ca4877e8d063baa1dd0b075fcd255fb6ab9e8a6bf98a276a2d6e8`.
The quantizer still targets SM90. The earlier `consumer_01` attempt ended with
35 passed and two identity-guard failures when FlashInfer added that auxiliary
environment key. Source bytes remained unchanged and no numerical mismatch
was reported, but that attempt is not accepted validation. The production
helper was not weakened or changed; the fresh run starts with stable settings.

`consumer_02.json` passed all 37 dense/grouped/packed MLP tests with zero
failures or skips (exec 8901, PID 1992312, exit 0). Result SHA-256 is
`cfd96e48bdc6a59c57d8a9b0374e3d3e1fefb8eb0dc92a1bf0a5a7a4980c11c7`.
Source and compiler identities match before/after; 25 actual loaded quantizer
specializations have matching retained CUBIN hashes and valid loaded handles.

`checkpoint_01.json` passed all nine actual-weight cases, layers 0–2 crossed
with Q121/128/1024, and 27 changed-input Graph replays (exec 48139,
PID 1993763, exit 0). Result SHA-256 is
`272a4917a304deeebfcf877434b52e62d444d28ada2a2dba3c54caafd7e1341b`.
Complete down-projection outputs are byte-exact against the unpacked path;
input/weight preservation, nondefault-stream ordering and retained outputs
pass. Source/compiler identity remains unchanged and six loaded quantizer
specializations are verified. This is actual-checkpoint MLP correctness with
synthetic BF16 activations, not complete-model propagation or serving timing.
Both GPU 3 processes have exited and the correctness grant is released.
