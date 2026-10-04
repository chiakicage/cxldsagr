# C6 executable plan

- [x] Establish caller/lease/clock/scratch contracts and checked baseline.
- [x] Read and rerun the independent 20,000-state CPU allocator audit.
- [x] Add separate native union/classify, compact, publish, and final-map helpers;
  keep generic gather_host_records and no persistent scratch growth.
- [x] Add validated Python bindings; route only exact private H <= P requests.
- [x] Add full-state differentials for partial-free recall, ties, alternating
  owners, fragmented/nonmonotonic pages, candidates, all-hit without a residency
  certificate, empty/padding IDs, rollover, streams, and failed transactions.
- [x] Run CPU metadata/allocator tests with CUDA disabled while GPU is held.
- [x] After C5 is validated and root grants a new window, compile C6 and run
  targeted GPU differentials plus the existing cache/indexer/prefetch suites.
- [x] Measure checked versus C6 exact API on the same GPU and snapshot, including
  Q128/N65664/P65536, hit mixtures, small partial-free pools, and actual kernel
  activity/launches. Record source hashes, dependencies, repeats and output paths.
- [x] Freeze a measured winner or reject it with evidence; send integration scope
  and limitations to root for full-request acceptance before result publication.
