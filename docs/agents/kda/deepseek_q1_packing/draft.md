# Draft before implementation

Baseline `echo._paged_q1_logits` allocates page64 storage and copies full keys,
full scales, zeroes the partial page, then copies its valid keys and scales.
NSYS attributes 67.137 us to the six full-page strided copies over three layers.
The official SGLang SetKAndS kernel writes one token per CTA into an already
persistent paged cache. Here the public API receives fresh contiguous K/scales;
copying all tokens using that append-oriented launch is not the same operation.
Keep official computation and fuse only the local input-layout adaptation.

First expose the existing packing as a helper for the official bridge. Build a
separate single-kernel candidate, preserving bytes through integer loads and
stores. One CTA writes each page; no page table or schedule caching. Compare
against the fixed tensor-copy implementation, including partial pages and
graph replay with changed bytes. Validate full official MQA outputs as well.

Risks: uninitialized padding, loss of NaN payloads through conversions,
incomplete fingerprints, and a launch configuration that makes many small
pages slower. Measure the complete MQA wrapper, not only the packing kernel.
Profile baseline and candidate with NCU; NCU replay timings are diagnostic.
