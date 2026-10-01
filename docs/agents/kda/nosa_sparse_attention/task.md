# NOSA resident block sparse attention: KDA task contract

Optimize complete resident NOSA block sparse attention toward 40% useful MFU
on captured L0, L15 and L31. Only L31 reaches the target in the accepted
checkpoint. Follow the shared workload, measurement and publication rules in
[the KDA index](../README.md). This is the resident component; the
[offload component](../nosa_offload_attention/task.md) has a separate contract.

The primary shape is BF16, 65536 prefix + 1024 queries, 32 Q heads, 2 KV heads,
D128, 64-token blocks and full NOSA 33/64 selection with CIS. Useful attention
FLOPs are 68,190,994,432, excluding padded or masked work. The retained 989
TFLOPS denominator gives a 172.374 µs budget. Include union preparation, work
sorting, output/exception checks and numerical repair in complete attention.

## Semantic and measurement constraints

Keep each query's exact selection membership, CIS bias, causal mask and
numerical behavior. Do not change policy or loosen the original tolerances.
BF16 native FA3 dispatch requires matching K/V strides and existing alignment
conditions; FP16 and independent K/V strides retain supported native fallback.
Empty queries must launch nothing. Keep build/header/dependency provenance
and exact kernel-sequence attribution for every supported path.

Profile union construction, partial membership processing, nonfinite handling,
TMA waits and WGMMA utilization. Reuse or overlap work only when ownership,
barrier/descriptor lifetime and actual resource limits are proven. Validate
real captures and synthetic selections, including legal all-fallback inputs;
a natural-case helper win must not hide a fallback regression.

See [checkpoint.md](checkpoint.md), [implementation_plan.md](implementation_plan.md)
and the historical [investigation log](investigation_log.md).
