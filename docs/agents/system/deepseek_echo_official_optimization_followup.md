# Official ECHO comparison after MFU optimization

The updated user goal explicitly requires this comparison after completing the
motivation MFU work. Existing published official/local results and backing
outputs remain until replacement runs and audits pass.

## Current dependency boundary

`OfficialDeepSeekServingBackend` inherits the shared checkpoint block/model and
resource code. It uses `OfficialAttentionRunner` for the pinned upstream
indexer/top-k/cache pipeline. Norm, dense MLP packing/shared quantization and
the upcoming local activation quantizer therefore affect its common model
computation. These changes require a fresh official run and fresh HBM control;
copying new motivation numbers into the old official comparison is insufficient.

The official experiment currently rejects compute graphs. Its custom attention
runner directly calls projection/output, and enabling a flag without auditing
dispatch could bypass or misrepresent the official pipeline. The final
comparison must state each implementation's compute policy. If common compute
graphs are adapted for the official path, validate its actual official
indexer/top-k/cache execution and full outputs before measuring. Do not label
graph/eager differences as isolated offload or cache improvements.

## Required completion work

1. Finish motivation optimization and freeze the promoted implementation.
2. Audit shared-model changes against the official cache/runner adapter, run
   full checkpoint correctness and preserve its predeclared numerical policy.
   The official unordered top-k baseline has measured repeat variability;
   do not replace its acceptance with an unsupported bitwise-equality claim.
3. Rerun H=P65536, A128, chunk1024, NH16777216, sixteen users/two rounds,
   independent copied weights/state, all hidden and final logits. Keep the
   independent HBM numerical repeat and candidate transfer counters.
4. Run quietly with source/dependency/precision identity and complete memory,
   lifecycle and numerical audits. Keep official pinned sources unchanged.
5. Regenerate official reports and the local/official comparison with current
   source/run identities, explicit compute/indexer/selection differences and
   contemporaneous per-experiment HBM controls. Publish accepted replacements
   and remove superseded affected report/output artifacts in the same update.

The optional common compute callbacks are now implemented in the official
runner. CPU checks cover separate resident/fused official dispatch, normalized
and unnormalized inputs, eager/callback computation, ordering and failure exit.
The opt-in checkpoint mirror check now has eager and compute-graph variants,
including actual replay-count checks. These GPU variants have not run under
the changed source; the experiment still rejects graph configuration.
See the [graph adaptation plan](deepseek_echo_official_graph_plan.md).
No new official performance result exists for C8 or the native quantizer.
