# Direct strided MLA rotary output

Parent: accepted C9, 2026-10-04. This is an isolated adapter candidate;
production sources and accepted reports remain unchanged.

## Contract and draft

The C9 HBM cold capture shows 640 main-MLA rotary input packing copies and
640 copies into the final Q tail. In the last source-0 history chunk these
kernels take 22.655 and 22.688 microseconds. These are sampled GPU durations,
not a prediction of complete-request savings. The installed FlashInfer
`_apply_rope_pos_ids_cos_sin_cache` accepts independent three-dimensional
input/output token and head strides, already used by its public wrapper.

Retain that exact vendor kernel and trig values. Pass sliced Q/K inputs
directly; let Q output address the final `[Q,128,576]` tensor's last 64
features. Keep K output owned, latent BMM output and the KV concatenation
unchanged. This requires no new GPU arithmetic or third-party source changes.
New handwritten GPU kernels for other candidates remain Triton.

Risks: private dependency API, output alignment/stride assumptions, aliasing
and preservation of the prefilled latent Q region, changed-input graph
replays, asynchronous writeback ownership, arbitrary model reference shapes.
The first prototype is temporary and applies only to main MLA projection.

## Executable plan

1. In `/tmp/deepseek_rotary_layout_20261004/`, compare the complete current
   rotary adapter plus final Q-tail copy with a direct-stride adapter. Test
   BF16/FP16, Q=1/5/121/128/129/1024, normal and aligned offset strided storage, main interleaved D64,
   signed zero, nonfinite stress and random inputs. Require exact output bits,
   unchanged inputs and latent guards. Indexer partial rotation and a generic
   optional-output API are outside this candidate.
2. Check non-default stream event ordering and graph replay with changed
   inputs and positions; preserve earlier owned/cloned outputs. For the
   selected main-MLA shape, time the complete allocating API in randomized
   paired order after warmup, eager and graph callbacks with borrowed graph-owned outputs, Q128/Q1024.
   Include final tail copy in the baseline and equivalent final storage
   allocation/ownership in both variants. Record all samples and identities.
3. If this screen passes, compare actual source layers 0–2 complete projection
   tensors for Q121/Q128/Q1024 and long positions, including changed-input
   graph replays. Compare complete projection callback timings in a separate
   confirmation window. Production integration requires independent review.
4. After promotion, validate integrated eager/graph outputs and affected
   experiment paths against frozen C9, then obtain a new formal/profile pair.
   Neither component timing nor source support replaces that acceptance.

Commands use `.venv/bin/python /tmp/deepseek_rotary_layout_20261004/probe.py`
with `screen` then `benchmark`, and later `projection`. All GPU work is
root-scheduled; pin the accepted SM90 and auxiliary PTXAS paths before imports.
No `torch.compile` is used. Artifact identities include the adapter, probe,
installed FlashInfer Python/native/header sources and loaded native module.
