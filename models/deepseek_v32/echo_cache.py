"""Scoped ECHO cache correction for GR requests that release candidate suffixes."""

from __future__ import annotations

import os
from contextlib import contextmanager

CACHE_ADAPTATION_ID = "gr_extend_recall_free_slots_v1"


def make_extend_recall_with_free_slots(host_module, torch_module):
    """Keep ECHO's extend recall flow, evicting only the allocation deficit.

    The serving adapter must bound each request's complete visible context by
    device-pool capacity, so the union of protected hits and misses fits. Device
    assertions guard the dynamic counts before eviction/allocation without a
    CPU scalar read. They are fatal CUDA assertions if an invariant is broken.
    """

    def recall_miss_tokens_extend_cuda_graph(
        self, recall_host_indices, device_page_table_1, layer_id
    ):
        self.host_need_recall.fill_(0)
        host_module.mark_host_need_recall(self.host_need_recall, recall_host_indices)
        recall_size = self.host_need_recall[:-1].sum().to(torch_module.int32)

        device_pool_allocator = self.device_pool_allocator[layer_id - self.start_layer]
        available_size = device_pool_allocator.available_size().to(torch_module.int32)
        pool_size = device_pool_allocator.size_in_tensor.to(torch_module.int32)
        valid_counts = (
            (available_size >= 0)
            & (available_size <= pool_size)
            & (recall_size >= 0)
            & (recall_size <= pool_size)
        )
        torch_module._assert_async(
            valid_counts, "ECHO GR recall counts exceed device-pool capacity"
        )
        free_size = torch_module.clamp_min(recall_size - available_size, 0)
        if os.getenv("EXTEND_OFFLOAD_DEBUG"):
            torch_module._assert_async(
                free_size <= pool_size - available_size,
                "ECHO GR recall eviction deficit exceeds occupied device slots",
            )

        self.free_device_pool_cuda_graph(free_size, layer_id, device_page_table_1)
        recall_device_indices = self.device_pool_loc_alloc_buf
        device_pool_allocator.alloc(recall_size, recall_device_indices)
        self.update_priority_when_use(layer_id, recall_device_indices)
        host_module.recall_update_extend(
            self.host_need_recall[:-1],
            recall_device_indices,
            self.host_token_to_device[layer_id - self.start_layer],
            self.device_token_to_host[layer_id - self.start_layer],
            self.device_pool.get_key_buffer(layer_id),
            self.kv_buffer[layer_id - self.start_layer],
            self.recall_counter,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
        )

    return recall_miss_tokens_extend_cuda_graph


@contextmanager
def scoped_extend_recall_fix(host_module, torch_module):
    pool_class = host_module.NSATokenToKVPoolHost
    original = pool_class.recall_miss_tokens_extend_cuda_graph
    pool_class.recall_miss_tokens_extend_cuda_graph = make_extend_recall_with_free_slots(
        host_module, torch_module
    )
    try:
        yield CACHE_ADAPTATION_ID
    finally:
        pool_class.recall_miss_tokens_extend_cuda_graph = original
