# Zero versus two external event nodes

The two earlier controls did not reproduce the formal no-event graph's idle:
collection state during construction and formal graph inspection each leave
the graph containing two external events near 16–20 us idle. Isolate the
remaining event-node difference on the same current CUB production source.

Run `q1_event_boundary.py` on GPU1/CPU8–15 with H=65,536/A=1/token111090,
three checkpoint layers, HBM cache, `return_hidden=False`, identical
FullExtendGraphCapture/CUPTI provider and collection settings. Both construct
under collection, then receive five stopped-collection graph warmups and one
in-trace replay warmup. Use independent model/cache/graph instances.

`no_events` adds no nodes to the production graph. `events` adds the same
begin/end `torch.cuda.Event(external=True)` records used by the earlier
controls. Do not time one arm using those graph nodes. Both arms use identical
ordinary CUDA event records around the model's `extend_graph_replay_q_1`
scope, bracketing the graph replay submission outside capture. This excludes
input staging and post-replay host commit from the CUDA interval while wall
time includes the complete synchronized forward. No outside timing event may
become a node in either graph.

1. Independently check both arms against eager and each other bitwise,
   including restored prefix and changed-token replay. Verify native node
   types differ by exactly two type-7 events and both contain 197 GPU nodes
   and 192 layer-owned GPU nodes. Bind current source/native/provider and
   input bytes in a fresh immutable receipt.
2. Require the receipt in a fresh no-NSYS process; retain 50 balanced AB/BA
   pairs of identical outside-CUDA-event intervals and synchronized wall time.
   Explicit CUPTI loading remains part of the stated process condition.
3. Collect a separate three-interval NSYS run: no-event construction,
   event construction, matched warmups and formal AB/BA replay. Verify both
   native ownership ledgers against same-process lineage and executable IDs.
4. Require all 197 GPU activities, kernel names/17 launch fields and D2D
   bytes to match across arms. Verify the no-event graph's GPU signature
   against root's current formal result. Compute the 192-node L0–L2 interval
   using native layer ownership, preserve boundary-crossing final norm and
   require equality of the full clipped-activity and selected-node unions.
5. Separate actual no-NSYS replay timing from recorded NSYS gaps. A profile
   gap change alone is not a production optimization or proof of the true
   hardware scheduling gap. Root owns any measurement correction/publication.

Use fresh check, benchmark, profile and analysis IDs. Prior control source and
artifacts remain read-only. Do not change production or introduce extra
capture-time edge enumeration into the no-event arm.
