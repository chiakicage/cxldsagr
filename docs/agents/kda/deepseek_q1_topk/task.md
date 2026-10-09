# Q1 deterministic top-k postprocessing contract

Optimize complete DeepSeek Q1 exact top-k on Hopper GPU1, CPU affinity 8–15.
Keep production `operators/deepseek_v32/indexer/selection.py` frozen. Retain
the exact installed FlashInfer FilteredTopKUnified SMALL selection kernel and
replace only its deterministic sorting/nonfinite-mask postprocessing in an
independent candidate. No precision, tie, ordering, invalid-ID or cache
semantics may be relaxed.

Input is inference FP32 `[Q,N]` causal scores, primarily Q=1, logical
N=65,537 or its 65,792-column aligned view, k=2,048. The API returns descending
FP32 values and int32 IDs, ties by smaller logical index under upstream radix
float ordering; every nonfinite value has ID -1. Values must match bitwise,
including signed zero and -inf. Arbitrary NaN model logits remain outside the
production contract. Check k in [1,2048], strided views, multiple rows and
changed-input graph replay even though promotion initially targets Q1.

The benchmark measures one complete exact-top-k API graph replay per sample,
including selection, deterministic ordering and nonfinite-ID masking. Score
generation, correctness, graph setup and source/native hashing are excluded.
Use balanced AB/BA order and retain every sample. Bind all input bytes,
candidate/baseline source, FlashInfer native library and actual compiled
candidate artifacts (Triton PTX/CUBIN or the CUB native library).

Promotion requires all bitwise checks, unchanged Filtered native identity,
consistent positive complete-API paired medians on real L0–L2 scores, and
independent NSYS/NCU explanation. Send a reviewed integration patch to root;
root owns production integration, full-model remeasurement and publication.
