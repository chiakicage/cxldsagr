# NOSA budget-serving execution checkpoint

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

This is engineering execution evidence, not the published experiment report.
Commands and measurement limits are in
[the selected-source handoff](nosa_gr_selected_source_commands.md).
Root authorized the complete GPU0/CPUs0–7/NUMA0/register8 sequence on
2026-10-04 after the fresh fixed full32 gate. Root independently ran the fixed
experiment on GPU5/CPUs48–55/NUMA1. External receipts retain this concurrency;
separate placement does not exclude shared storage, driver or power effects.

## Correctness and source identity

The four selected shared HBM/dense tests passed without skips. The existing
budget checkpoint test then passed H65536/A128 over all 32 checkpoint layers,
four schemes, two users and two visits: all 16 candidate hidden outputs were
exact. These are correctness checks, not serving latency measurements. Their
launch/source/placement records remain under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_gpu0_execution_20261004_01/`.

The first H4K formal attempt completed 804 request rows but failed the report
validator: it treated a reserved, lazily allocated HBM host flag as already
allocated. H4K and H16K do not reach the checked-indexer length that creates
the flag. Actual zero host-flag bytes are valid; the conservative reservation
remains unchanged. The failed `_01` data remain external at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/gr-serving-gr_nosa_flags_h4k_20261004_01.TpL70Q/`
and are excluded from results. No profile was attached to that failed run.

Root reviewed the narrow correction to `report.py`, `audit.py` and
`test_nosa_shared_accounting.py`. It passed 167 focused CPU tests and a
read-only check of the failed rows; that check did not promote the failed run.
The new 280-file GR source digest is
`8dc00e23469a67ebee67eb603859c9de9987e9151520a607d1bb20a498e547f9`.
The unchanged 114-file fixed-experiment source is
`a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`.
The new freeze and the retained 444-request identity proof are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_selected_source_20261004_02/`.
The unchanged runtime GPU gates were not repeated for this reporting change.

## Current family status

All formal runs retain users 1/8/32/64/128/256/512, A128/C1024, seed42,
Beauty interaction-count heat, max-revisits8, two warmups, exact hidden
comparison and HBM 4 GiB/DRAM 16 GiB budgets. H4K/H16K each have 201 requests
per scheme; H64K has 42. Every profile retains three real revisits and the
full-batch union-once serialized control. No result is discarded for failing
the 90% overlap claim.

| Family | Formal run | Matching profile | Independent CPU review |
| --- | --- | --- | --- |
| H4K | `gr_nosa_flags_h4k_20261004_02`: accepted, 804 rows/28 cases, 603 exact non-HBM comparisons | `gr_nosa_flags_h4k_profile_20261004_02`: accepted; request IDs 2/3/7 | Passed; receipt SHA256 `51447d78645c0e921e27fd2049b43221c36a87bd01282f96c11ed63dce6310ed` |
| H16K | `gr_nosa_flags_h16k_20261004_02`: accepted, 804 rows/28 cases, 603 exact non-HBM comparisons | `gr_nosa_flags_h16k_profile_20261004_02`: accepted; request IDs 2/3/7 | Passed; receipt SHA256 `e9a82394a25df78fbfcd4bcc5ee1ee864e1a1aab4b53e5c28796fa20c1e2f0bc` |
| H64K | `gr_nosa_flags_h64k_20261004_02`: accepted, 168 rows/28 cases, 126 exact non-HBM comparisons | `gr_nosa_flags_h64k_profile_20261004_02`: accepted; request IDs 1/2/3 | Passed; receipt SHA256 `6fc9d705eee98452a99794d1c2f7edb5f582df5db42150a85bbb1b76d8c4c9ab` |

H4K page-envelope and nonempty-stripe overlap ratios are equal in each sample:
0.6635573122529644, 0.691590287827625 and 0.6905228758169935. All three
samples fail the 90% claim; the valid observations remain retained. These
softmax-overlap metrics do not replace complete request latency. H16K also
passed measurement validity; its three equal page/stripe ratios are
0.7650770663080584, 0.7592333799161509 and 0.755853165044802, all below90%.
H64K passed the same gate with equal page/stripe ratios
0.7602926829268293, 0.7421515696860628 and 0.7706589706589707.
All nine overlap samples remain valid despite failing the 90% claim.
The H64K independent launch/provenance receipt SHA256 is
`c50060d153790b65d6a2edcefc3c15919a0dda2b8c970c9a093cfdb1c092b997`.

The H4K independent review reopened saved outputs, checked all seven complete
request files against retained originals and the existing proof, validated
profile selection/reference linkage, and rehashed both source snapshots.
Its receipt is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_independent_audit_20261004_02/h4k_family_review.json`.

New stage receipts are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_gpu0_execution_20261004_02/`.
They record exact commands/environment, inherited bind0 policy, observed
target affinity and raw `numa_maps`, with source checks before and after each
stage. Formal metadata establishes the target GPU UUID
`1bdee8b4-22ac-536c-208b-bfb4ed38b878`, SM90 and eight CPU threads. These
records do not prove blanket node0 physical residency, full physical HBM
capacity, or an allocator provider that the target process did not record.

Root accepted the complete replacement and authorized publication on
2026-10-04. The new README and all three report families are now installed;
the exact superseded NOSA report/output inventory was cleaned only after
installed hashes and links passed. No fresh or unrelated output was deleted.


## External publication preparation

All GPU stages and independent family/provenance audits are complete. The
external replacement package is at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_publication_stage_20261004_02/`.
It contains the polished Chinese README, 57 selected report assets, the
historical-review navigation clarification, reproduction scripts/data, and
the hash-guarded original NOSA removal inventory. The proposed report
distinguishes sampled HBM tensor storage, owned pinned DRAM bins,
active-allocation reservations and PyTorch allocated/reserved.

The aggregate observations are HBM's lowest all-request mean in all 21
history/population combinations, overlap's lower all-request mean in 19/21
versus serial sparse, and lower revisit mean in 2/17 nonempty combinations.
H16K has 13 HBM revisit misses and no offload revisit miss; H4K/H64K have no
revisit misses for any scheme. Ledger session capacities are 29/7/1 for HBM
and 63/15/3 for each offload scheme. These do not establish physical HBM
capacity or unmeasured population scalability.

Staging checks passed all 66 README links, executable shell-block syntax,
copied/generated identities and a fresh rehash of the complete old allowlist.
The publishing agent inspected all 12 summary panels and 42 request panels.
Independent report/table review passed, followed by a narrow review of the
two entrypoint-navigation updates. Root then authorized installation and
guarded cleanup. The installed package contains 57 report assets, the README,
the historical navigation clarification and reproduction files. All 63 mapped
file hashes and 66 README links passed before cleanup. Publication removed
12 stale report files, replaced 45 old report paths with the reviewed assets,
and deleted exactly 18 old output category directories containing 1638 files
and 623125320 bytes. The six fresh formal/profile families remain accepted.


## Published installation and cleanup evidence

The publication plan SHA256 is
`1e1f509abf34f4ed112bb7ca3931cfb2a6aa0715834490eec5b4f169cb6b7de6`.
The complete independent report/table review is
`publication_stage_review.json` with SHA256
`4072504541fb34bef7aac3cfce00bda1d826652103068167f316b4d505a9f392`;
the navigation delta review is
`publication_navigation_delta_review.json` with SHA256
`6495742784f410d08c51174a0bf3faa0d7f25dd9cfbe3a4af2dc20aa9cffbe04`.
Both reviews remain under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_independent_audit_20261004_02/`.

The install/cleanup receipt is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_gr_publication_stage_20261004_02/publication_receipt.json`
with SHA256
`6e7c84cba390ec78cf250fa03b8bd74fbe589e18c7ba0e85959d4c3b07537aed`.
It records the exact installed hashes, reviewed narrow root/index changes,
pre-cleanup verification, removed files and preserved fresh runs. Its copies,
the plan, old inventory, independent reviews and publishing helper are retained
under
`experiments/gr_serving/output/data/nosa_gr_publication_20261004_02/`.
Published family provenance files preserve their reviewed preparation identity;
the final receipt closes publication and cleanup. Runtime code was not changed.


The independent postpublication check passed all 63 installed hashes,
the exact 57-report-file set, 66 README links, 12 stale report removals,
18 old output directory removals and all six accepted fresh families.
Reversing each localized root/index phrase change reproduced its before hash,
so unrelated navigation text was preserved. The receipt is
`postpublication_review.json` in the same independent-audit directory,
SHA256
`0631deb7c04e3c4d00cdf34e81e717a196518efced7054302b6ad71e076a9441`, with a copy in the publication output directory.
This check did not repeat tensors, tables or GPU execution. A final source
freeze check also passed for GR 8dc00e23 and fixed a73ad32b, without CUDA
initialization.
