# Q1 page64 packing

Reduce the local HBM preparation cost on SM90 without replacing the official
DeepGEMM MQA or ECHO paged decode kernels. The current H65536/A1 trace contains
six strided key/scale copies totalling 67.137 us over three layers. The official
reference L0–L2 window is 1.014975 ms and the local accepted window is 1.265187 ms;
these are different token/environment workloads, not an equivalence claim.

Input: contiguous FP8 E4M3 K[N,128] and FP32 scales[N]. Output: uint8
[ceil(N/64),8448], with 8192 key bytes followed by 256 scale bytes per page.
Every input bit, including signed zero and NaN payloads, must be preserved;
all page padding must be zero. No persistent cache or hidden preparation.

Candidate: one coalesced packing kernel before the unchanged official paged
MQA. GPU 0, CPUs 0–7. Q1 H64/D128 and real L0–L2 inputs are the main acceptance
scope. Validate page boundaries, changed graph inputs, raw scores and exact
top-k before independent timing. Keep all timing samples and source/native
identities. A component win permits integration, not a full-model claim.

Commands use `CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 .venv/bin/python -B -m
experiments.deepseek_v32_echo_official.src.q1_packing` with `--mode check`,
`--mode bench --receipt <accepted-result>`, or `--mode profile`. Checks go to
`/tmp/cxldsagr-checks/deepseek_q1_packing/<run_id>`; performance/profile data
use the experiment output directories. Full-model promotion additionally
requires fresh independent MFU check, bench, profile and capacity accounting.
