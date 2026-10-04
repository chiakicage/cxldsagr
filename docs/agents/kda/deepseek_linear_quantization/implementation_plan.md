# Native linear quantization executable plan

- [x] Freeze C8 reference and read pinned helper, native FFI adapter and the
  independent [semantics review](semantics_review.md).
- [x] Write task and draft before native implementation.
- [x] Build temporary CUDA C++/TVM FFI candidate `warp1_cta4` plus isolated
  official-oracle driver. Keep production source unchanged.
- [x] Run CPU-safe import/build-info tests; inspect explicit SM90 offline build,
  emitted PTX and actual source/include dependency identity. Confirm no CUDA
  context initialization and no production Triton/torch.compile dependency.
- [x] Obtain independent source/harness review, then root's GPU correctness grant.
- [x] Exact screen all dtypes, layouts, tails, nonfinite/rounding cases, output
  ownership, nondefault streams and changed-input graphs. Compare FP8 uint8
  and scale uint32 views; preserve input/backing/guard bytes. Save every case.
- [x] Independently audit screen record, generated oracle code and source hashes:
  115 exact fixtures and six lifecycle cases; see [checkpoint.md](checkpoint.md).
- [ ] Add one geometry/vectorization candidate at a time, recording parent and
  source identity; repeat affected exactness before timing it.
- [ ] After exclusive timing grant, compare complete API costs with interleaved
  ordering. Measure every primary M/K pair, then confirm a selected winner in
  a separate granted window with a different ordering seed.
- [ ] Integrate only an accepted native adapter; preserve CPU reference, empty
  behavior, non-SM90 rejection, contiguous output layout and public signature.
  Root updates backend provenance to remove compiled-production metadata.
- [ ] Run focused dense/grouped/packed MLP correctness, actual checkpoint and
  graph/stream lifetime checks under production source identity.
- [ ] Root runs combined four-scheme H64K correctness, matching full-serving
  and operator MFU measurement, then publishes and replaces affected reports.

## Screen matrix and exactness

Use M=0,1,2,7,8,15,16,17,31,32,33,63,64,65,121,128,129,1024 in a bounded
covering matrix. K=1,24,127,128,129,257,1025,1536,2048,7167,7168,7169,16384,
18432. Include contiguous, padded-row, offset, column-stride, transposed and
zero-stride read-only inputs. Fill excluded backing positions with NaN/large
values and compare the full owner before/after. Invalid dtypes/ranks/K and
non-SM90 behavior belong to wrapper validation.

BF16 and FP16 screens cover all 65536 encodings, plus independently controlled
group maxima. FP32 includes adjacent values around `448*2^e`, floor boundaries,
normal/subnormal limits, and FP8 halfway cases with a fixed separate maximum.
Place positive/negative Inf, signed/payload NaNs and signed zero at several
reduction lanes and in partial groups. Verify every byte of data and scales.

Keep an uninterrupted varied native-call sequence to expose cache/dispatch
state errors. Compiled-oracle case isolation may reset Dynamo only between
independent cases, never within a stream/graph ownership case. Save oracle
source hashes and generated code identity; historical PTX is supporting source
evidence, not the identity of a new run.

Retain outputs across later native calls and allocator churn. Warm before graph
capture; use independent graph pools, changed static inputs, borrowed replay
identity and owned cloned outputs retained across later replays. A nondefault
stream case must contain a predecessor and a waited downstream consumer.

## Reproducible commands

The temporary driver paths are fixed below. `prepare` and `screen` are executable;
the separate benchmark driver preserves the accepted screen driver's bytes.
Every new run uses a fresh output name.
Root supplies the granted physical GPU and exclusive timing interval.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python /tmp/deepseek_linear_native_quantization_20261004/driver.py \
  prepare --output /tmp/deepseek_linear_native_quantization_20261004/prepare_01.json

PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=<granted_gpu> OMP_NUM_THREADS=8 \
  .venv/bin/python /tmp/deepseek_linear_native_quantization_20261004/driver.py \
  screen --prepare /tmp/deepseek_linear_native_quantization_20261004/prepare_01.json \
  --output /tmp/deepseek_linear_native_quantization_20261004/screen_01.json

PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=<granted_gpu> OMP_NUM_THREADS=8 \
  .venv/bin/python /tmp/deepseek_linear_native_quantization_20261004/benchmark.py \
  --screen /tmp/deepseek_linear_native_quantization_20261004/screen_01.json \
  --warmups 10 --repeats 50 --seed 20261004 \
  --output /tmp/deepseek_linear_native_quantization_20261004/bench_01.json
```

## Measurement and promotion

Record device/clock state, compiler and dependency versions, source/binary/PTX
hashes, variant dispatch, complete samples, warmups/repeats, ordering seed and
pre/post exactness. Primary BF16 shapes are M=1,121,128,1024 crossed with
K=1536,7168,16384,18432. Add saved real checkpoint source-0/1/2 activations
where available, clearly distinguishing them from synthetic inputs.

Eager CPU wall completion and joined CUDA events are separately sampled and
both include public allocation and launch/layout work. Graph replay-only time
is diagnostic; a complete graph boundary includes static-input copy, replay,
owned output allocation/clone and joined completion. Graph capture/setup and
pool memory are reported separately. Alternate candidate/oracle ordering and
inspect correctness outside samples. Do not subtract independent timing series.

Promotion requires zero byte mismatches or ownership/stream/guard failures,
unchanged source through execution, and a reproducible complete-API improvement
on the actual target shapes without a material regression requiring hidden
fallbacks. Review small-M overhead separately. A noisy or regressing variant is
revised or rejected with its evidence preserved in the component ledger.
Integration must rerun the actual production adapters: prototype timing alone
does not demonstrate packed MLP, serving latency or measured MFU improvement.
