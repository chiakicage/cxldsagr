# Consumer-layout scales: private model integration

The component's exactness check, two fixed timing windows and profile audit have
passed. Evidence is recorded in the active DeepSeek MFU task. Q1 regressions and
Q1024 borrowed-replay order reversals remain part of that evidence. Only Q128 and
Q1024 ordinary FP8 linears may opt in; there is no unrestricted promotion.
Cache-manager GPU checks and screens retain execution priority.

## Integration draft and scope

Use the frozen numerical sources at
`/tmp/deepseek_linear_scale_layout_candidate_20261006_03/candidate/linear/`.
A new private adapter replaces only the known checkpoint-linear objects in a
diagnostic three-layer model, before graph preparation. It keeps the original
linear object and borrows its weight/scales without allocating or copying tensor
storage. The adapter selects the frozen consumer-layout implementation only for
BF16 CUDA matrices with Q128/Q1024 and a verified component K/N pair. All other
inputs delegate to the original object. No global production function is patched.

The six projection linears and three dense-MLP linears in each layer are explicit
targets. Indexer score quantization, grouped/MoE, top-k, cache, attention and model
scheduling remain their original calls. Existing prepared activation/output
contracts stay in the selected complete linear implementation; adjacent producer
and consumer must use the same implementation. A private prepared object passed
to an unsupported consumer is rejected, never silently converted or consumed.

Bind exactly the diagnostic model before any graph bank exists. Validate all
targets first, then apply replacements transactionally; preserve primary and any
restoration errors if binding fails. The original objects remain owned by the
adapter for the model's lifetime. Unbinding or switching a captured model is not
an API: baseline and candidate use separately prepared graph banks. No new
capacity claim follows from borrowing weights; graph-private allocation and
reserved capacity still require actual model measurement.

## Executable gates

1. Write private adapter and loader under
   `/tmp/deepseek_scale_layout_model_adapter_20261006_01/`. Bind the frozen source
   manifest and verify the production linear baseline before import. Keep all
   numerical sources and existing experiments unchanged.
2. CPU checks cover the exact target set, original weight/storage ownership,
   default/unsupported dispatch, adjacent prepared producer/consumer dispatch,
   malformed/duplicate model binding and transactional restoration. These are
   integration checks, not numerical or GPU acceptance.
3. Independently review the private adapter before root GPU use. Reuse the
   accepted full-model check/screen infrastructure with this sole intervention.
   Capture fresh Q128/Q1024 graphs, preserve unchanged input/precision/cache paths,
   save complete hidden/logits/top-k/cache before/after state, verify the actual
   candidate branches during setup, and bind actual quantizer/native artifacts.
4. Run fresh fixed four-method prefill/cold/warm paired timing only after the
   model check passes. Keep all samples and order strata, including regressions;
   complete-call timing decides whether to proceed. Never infer model benefit
   from the component or subtract control cells.
5. Only an accepted model timing result permits a fresh labeled complete/layer
   gap profile. The original strict full-extend and every-layer 10% gates and
   complete prefill absolute-gap ratio <=1.2 still apply. Formal MFU/motivation
   publication remains downstream of those goals. Production stays unchanged
   until independent integration and experiment requirements are satisfied.

This plan does not authorize an unchanged component remeasurement or claim that
model performance improves. The initial work is source/CPU preparation while
cache-manager candidates complete their higher-priority execution gates.
