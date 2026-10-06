# DeepSeek static capacity refresh

The five offline plans completed on 2026-10-06 with `CUDA_VISIBLE_DEVICES=''`.
They use the same parameters, checkpoint config and supplied hardware memory
inputs as their corresponding `20261005_02` plans. No model or GPU execution was
performed. Existing published reports remain in place until the new complete
request memory evidence passes the unified report audit.

| New run ID | P | NH | U | HBM total bytes | HBM remaining bytes | Next candidate fails |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `cache_deepseek_nh_20261006_01` | 32,768 | 29,818,880 | 455 | 48,944,687,168 | 76,131,928,819 | DRAM budget |
| `cache_deepseek_p_base_20261006_01` | 10,296,447 | 1,048,576 | 16 | 125,076,590,080 | 25,907 | HBM budget |
| `cache_deepseek_p_base_headroom_20261006_01` | 8,983,043 | 1,048,576 | 16 | 125,076,608,768 | 7,219 | HBM budget |
| `cache_deepseek_p_nhmax_20261006_01` | 6,451,364 | 29,818,880 | 455 | 125,076,610,368 | 5,619 | HBM budget |
| `cache_deepseek_p_nhmax_headroom_20261006_01` | 5,137,960 | 29,818,880 | 455 | 125,076,611,648 | 4,339 | HBM budget |

The two headroom cases include an explicitly selected 14.5 GiB deduction,
15,569,256,448 bytes. It is an old observed allocator/device difference reused
as a planning assumption, not preallocated storage or a validated current
requirement. The total includes this deduction. All five P/NH maxima and rounded
HBM totals are unchanged.

The eighth per-layer int64 counter adds 80 logical bytes per ten-layer session.
The combined counter slab grows from 560 to 640 bytes, within the same allocator
block. Logical HBM reservation therefore rises by 1,280 bytes for U=16 or 36,400
bytes for U=455; the allocator allowance falls by the same amount. The static
ledger's `session_native_counters` is now 40 bytes per layer/session; prefetch
counters remain 24 bytes per layer/session.

## Audit evidence

All artifacts are under
`experiments/cache_management/output/data/cache_deepseek_plan_refresh_20261006_01/`:

- `summary.json`, `plans.csv`: selected points, totals, remaining budgets and deltas.
- `audit.json`: independent verification of all 60 archived source files, five
  checkpoint config hashes, historical input equality and selected/next-candidate
  boundary checks through `capacity_report.read_plan(path, [])`.
- `deltas.json`: all old/new plan differences, including next candidates and useful
  selected points. Only native counters, logical reservation and allocator
  allowances changed.
- `plan_commands.json`: exact five invocation records, CPU-only environment,
  elapsed times and zero exit status. Each original run also saves `command.json`.
- `audit_helpers.json` and `source/`: identities and snapshots of audit/planning
  helpers. CUDA remained uninitialized during this independent audit.

All five plan source manifests have aggregate SHA-256
`1c6fbf6d1bb0f164ed9d599c4ec40e03d8b174010199ce597a29b3af977029aa`.
The checkpoint config SHA-256 is
`c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1`.

## Pending publication edits

After the new local C10 bench and unified report are accepted, update the cache
README's DeepSeek ledger values:

- Session hints plus prefetch/native counters: `2,480 B` becomes `2,560 B`.
- Per-session private HBM: `86,514,096 B` becomes `86,514,176 B`.
- `HBM_base` uses `86,514,176U` instead of `86,514,096U`.
- Replace the five old plan IDs with the new IDs above; P/NH table values stay
  unchanged.
- Replace complete-request memory tables only from the newly accepted C10 bench
  and unified report, including new dense ticket and graph accounting. The old
  static ECHO/serial history-only scope does not acquire dense-specific storage.

The exact future unified report invocation and all C10 check/bench/profile
commands are in [c10_commands.md](c10_commands.md). The report command has not
been run against old DeepSeek request observations. No report or README was
replaced by this static refresh.
