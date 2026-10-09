# Promotion draft

The baseline serializes 64 record copies inside one 256-thread CTA. Its shared
metadata checks and 2-byte copies preserve all BF16 alignments. A naive grid
expansion would invalidate shared state and may race validation against mapping
publication. Use two stream-ordered phases: first validate all IDs/slots/owners
and duplicates with one CTA, then copy and publish one record per CTA. Validation
finishes before any pool mutation. Valid owners and staged hosts are disjoint
because their forward-map encodings differ. Distinct slots/hosts make the second
phase's writes disjoint; each CTA synchronizes its copy before publication.

Start with 128 threads per record CTA, no vector alignment assumption and no new
scratch. Validation computes exact statistics; copy/publish does not touch them.
If the extra launch outweighs the copy gain, revise only after measured evidence.
Do not infer benefit or overlap from launch structure.

Check uses CPU reference and baseline exact comparison plus existing adapter
cases. Bench restores identical inputs outside timed graph replay, alternates
baseline/candidate order, and uses GPU events around the whole promotion call.
Use actual layer-2 saved BF16 KV for clean timing; random 16-bit patterns cover
NaNs and payload preservation in correctness checks. One-call profile mode
excludes preparation/validation comparison from the capture window.
