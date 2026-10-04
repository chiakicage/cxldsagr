"""Offline module MFU from correlated nsys kernel durations and NOSA NVTX scopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from experiments.nosa_baseline_performance.src.sparse.analyze import (
    FUSED_STAGES,
    NESTED_STAGES,
    PHASES,
    PRIMARY_STAGES,
    STAGES,
    _integer,
)
from experiments.nosa_baseline_performance.src.sparse.mfu import (
    NATIVE_INDEXER_REVISIONS,
    build_report,
    matrix_flops,
    validate_kernel_backend,
)

ROOT_RANGE = re.compile(r"NOSA/profile/(full_prefill|extend)/(\d+)\Z")
STAGE_RANGE = re.compile(r"NOSA/(full_prefill|extend)/layer_(\d+)/([a-z_]+)/q(\d+)\+(\d+)\Z")
LINEARS = ("qkv_proj", "cis_projection", "o_proj", "gate_up_proj", "down_proj")
MODULES = (
    *LINEARS,
    *[
        stage
        for stage in STAGES
        if stage not in ("cis_projection", "native_indexer", "native_checked_indexer")
    ],
    "cis_projection_gemm",
    "score_selection",
    "native_finite_check",
    "native_ranked_compression",
    "other_non_matrix",
)
PARENTS = {stage: "indexer_total" for stage in NESTED_STAGES} | {
    "cis_projection_gemm": "cis_projection",
    "score_selection": "indexer_total",
    "native_finite_check": "indexer_total",
    "native_ranked_compression": "indexer_total",
}
QK_MODULES = ("compressed_scores", "pooled_scores", "score_selection")
INCLUSIVE_MODULES = {*STAGES, "score_selection", "native_finite_check", "native_ranked_compression"}
FUSED_SCORE_FAMILY = "nosa_scores::fused_scores::fused_scores_kernel"
PRUNED_SCORE_FAMILY = "nosa_scores::pruned_scores::fused_scores_kernel"
FUSED_SCORE_FAMILIES = (FUSED_SCORE_FAMILY, PRUNED_SCORE_FAMILY)
FA3_ATTENTION_SEQUENCE = (
    "nosa_fa3::prepare",
    "flashinfer::PrefillWithKVCacheKernel",
    "nosa_attention::attention_kernel",
)
FA3_SORTED_ATTENTION_SEQUENCE = (
    FA3_ATTENTION_SEQUENCE[0],
    "nosa_fa3::sort_work_by_union_size",
    *FA3_ATTENTION_SEQUENCE[1:],
)
FA3_MAIN_TEMPLATE = re.compile(
    r"flashinfer::PrefillWithKVCacheKernel\s*<\s*nosa_fa3::ML\s*,\s*"
    r"nosa_fa3::EP\s*,\s*nosa_fa3::KT\s*,\s*(?:\(bool\)\s*)?(?:0|false)\s*,\s*"
    r"(?:\(bool\)\s*)?(?:0|false)\s*,\s*nosa_fa3::Scheduler\s*,\s*"
    r"(?:\(bool\)\s*)?(?:0|false)\s*>"
)
# These inference graphs share the reviewed unscoped GEMM order. Keep complete
# versions so an old captured snapshot remains analyzable after optimization.
_LEGACY_GRAPH = {
    "models/nosa/model.py": "5acbd9c5349f75b5b02366849ef7cd76d6f5b0676126bc789b8384939dfc51a5",
    "models/nosa/layers.py": "cf54c7c1e96483175da788491450bd264fae322a6c1452f830b5ce721f308b43",
    "models/nosa/scoring.py": "c4c7e75a15a5ffb95e935b02ec482586a288c64ad079a175b293db6baa3f92d8",
    "models/nosa/indexer.py": "4b1ef076fd759ff6c6f114673fa357488e0ca1d4474a1b20b90430d3db6704c7",
    "models/nosa/attention.py": "1592ff0607cda99b94e75fa62ca4a64f4ec086be4e7aa4cd8714f610c6f07d84",
    "layers/feed_forward.py": "95dcebaa67b4f2069f580ac0831ef8335fc15b022e802708e24f58c02408de4a",
    "operators/sm90/nosa_indexer.py": "640cff9441efe24e579b64d7f91eb6684075f35ae32243f0c46a988b5b7453b5",
    "operators/sm90/_nosa_attention_triton.py": "c9b2060928cff8cbf594a876061546521064da78f52b3297922bcc5dd0f96985",
}
SUPPORTED_GRAPHS = (
    _LEGACY_GRAPH,
    _LEGACY_GRAPH
    | {
        "models/nosa/indexer.py": "cbaa654ddcd5db2de65f2894361bd277bec95911c60adf344099f0268dc8bba7",
        "operators/sm90/nosa_indexer.py": "bc402c60d3283d8651c562f41689f3e70842ce226b2e244c5db83437aa43fd4b",
    },
    _LEGACY_GRAPH
    | {
        "cache/manager.py": "3f8a3129900f8e5e1af6005fcc187f5d430c5ce5156ca80c169384694dd39f91",
        "cache/indexer_cache.py": "da80484c5d2ad016b3119f80576e72d2956f8d9d1a5cc2804a0e6d668b80b2eb",
        "models/nosa/cache.py": "aa8e5f9a71a4c715e22567e9dcaa0d88d795c9a76a127431e7eb2045658d2f1d",
        "models/nosa/indexer.py": "e5f636a626ca2319698fb28ce49c9637088bc3af4b98ba197feafbd72f165633",
        "operators/sm90/nosa_indexer.py": "980c667fc87d7d62b8165cf10b0385cfbefd8b81419f2b1f81f660014b6e78eb",
        "operators/sm90/nosa_compression.py": "9655a9b4caeea836c846eef654f781318fc930e5ee95192fbeb6ee64dc8201f3",
        "operators/sm90/nosa_validation.py": "aded63a061790919d0596c18f4129d4c77782b53c4e762ae87f7bd3aa2bb9700",
    },
)
# The migrated native kernels and the remeasured Triton control share this
# dispatcher/source tree. Native score scopes contain normalizer + score
# launches; this first native attention revision contains one WGMMA/TMA launch.
SUPPORTED_GRAPHS += (
    SUPPORTED_GRAPHS[-1]
    | {
        "operators/sm90/nosa_indexer.py": "5319079fcd2810d3e8d0437b15d32c3b4fee9c020436b5aaf6f7178baf3ceedd",
        "operators/sm90/nosa_attention.py": "69c587aa7e15afe3dc4d813bab1a6b01785b5c3db9d6dca5bfa7e68159d5563d",
        "operators/sm90/_native.py": "1a852683700af461fc2de5b12295678da86026b92b031b3b609aad6fa5af1ff3",
        "operators/sm90/_nosa_scores_cuda.py": "70a5cc50c3916675d7a23b9dca49e531a7bed9026e0f5f0bc254bd0a24bd170c",
        "operators/sm90/_nosa_attention_cuda.py": "52371a3e7441070f9dd124fca4b7664fc282a8022292fc36357e5ca11f63a9d3",
        "operators/sm90/csrc/nosa_scores.cu": "319c6a58c45e6e77fcf3b23d8d6e104f89c99d569e6d171f02412bc846b87830",
        "operators/sm90/csrc/nosa_attention.cu": "c669d9a3f1fa9be72cd2578e6bfeb797b765798e8498a1df154c932b120e514e",
    },
)

GROUPED_ATTENTION_SOURCE = "operators/sm90/csrc/nosa_attention_grouped.cuh"
# The grouped revision leaves the five model projections and their ordering
# unchanged. Each native attention call with >=4 query rows launches the
# grouped kernel and then its per-query fallback, even if no group falls back.
SUPPORTED_GRAPHS += (
    SUPPORTED_GRAPHS[-1]
    | {
        "operators/sm90/nosa_indexer.py": "1c34d2ebddba31cd2d06d1468c36535a9ab835ee28e134a6a8b4bf1e32df028a",
        "operators/sm90/nosa_attention.py": "0c1b422a5fe5c485927f017f4c2578271eaf6258d01cc3c1aa6d33d488d1d6f9",
        "operators/sm90/_nosa_attention_cuda.py": "66f737253c18dacf91eb066c2b07041864b91e7e7fa5f6e674943c424aca038e",
        "operators/sm90/csrc/nosa_scores.cu": "0897a00d6d38dccf88566771f063b71e8de16f9458d1f4f6e3bcc71ff91e84d2",
        "operators/sm90/csrc/nosa_attention.cu": "158aad48fc48b0319bea8de939483580f40115d069813958537b956d97a8fefa",
        GROUPED_ATTENTION_SOURCE: "21826cbc696c86de2d7fa5b8df185713e37827a8b80c909ad38bfd275383d8b9",
    },
)


# The accepted BF16-pair/FA3-v3 checkpoint retains the five model projections
# and their unscoped GEMM order. Pin all captured inference/cache/kernel sources.
# The second version only reformats the FA3 Python adapter (identical AST).
# Legacy graphs above remain available for their original published reports.
_BF16_PAIR_GRAPH = {
    "cache/contracts.py": "58575f1c9aee3f2843b7f2bce07f1589ea8cb9a712b050991199a95639f97f73",
    "cache/indexer_cache.py": "da80484c5d2ad016b3119f80576e72d2956f8d9d1a5cc2804a0e6d668b80b2eb",
    "cache/manager.py": "3f8a3129900f8e5e1af6005fcc187f5d430c5ce5156ca80c169384694dd39f91",
    "executor/model_executor.py": "f585f9940d5fb506e7f8aaeafe8602f9877ac773f2183603744f2b3a2a48c919",
    "layers/attention.py": "91eccef789f099017c229b99d1d0ccc9d66ce980ab6dbc772a25fe031f6f34d3",
    "layers/feed_forward.py": "95dcebaa67b4f2069f580ac0831ef8335fc15b022e802708e24f58c02408de4a",
    "layers/normalization.py": "426b2cd0e1da6c944ff661a93d55eee2b064585d5c5f6c6e6a0b80ca3bc98af0",
    "models/nosa/attention.py": "1592ff0607cda99b94e75fa62ca4a64f4ec086be4e7aa4cd8714f610c6f07d84",
    "models/nosa/cache.py": "4fa806fae9e4882086d6409c06f34706d5519a428fa4943cbf6ea2cd6cf10ed7",
    "models/nosa/config.py": "eaa168a439fada20f13b6d0a44c801de30e3b1c04d3e5ac43d4273a3454db165",
    "models/nosa/indexer.py": "480890dfa8744ff7ca88f90c1878673b57c98319b42da936b3b66f8da2d99bf7",
    "models/nosa/infer.py": "909541308008b155071db3af00927f4861b201ae7fc8e7c0bd9291ee1b19c672",
    "models/nosa/layers.py": "cf54c7c1e96483175da788491450bd264fae322a6c1452f830b5ce721f308b43",
    "models/nosa/model.py": "5acbd9c5349f75b5b02366849ef7cd76d6f5b0676126bc789b8384939dfc51a5",
    "models/nosa/request_format.py": "25c775ba0b76cab91663b827fc8b38fc20511895a5e6eff3054273cbfe418825",
    "models/nosa/rotary.py": "f14c898cd77a72bd4e6908ef59b0b5fe5d42b2ebadb34aa5db092577d154dbde",
    "models/nosa/scoring.py": "c4c7e75a15a5ffb95e935b02ec482586a288c64ad079a175b293db6baa3f92d8",
    "operators/flashinfer.py": "0573d16ce4dd3e13b3792c6b1a4b8c9bab1477ee0f7bfce2abfad87c4c305fd9",
    "operators/sm90/_native.py": "e7fc70b219647303b3295c38f536e1014ab52f4b0a0dcb7f8108b8c0ca2a8f26",
    "operators/sm90/_nosa_attention_cuda.py": "5a2c61297c343ce51f9227b48906fb4f77a1b730b1e3116815f9890255f84afa",
    "operators/sm90/_nosa_attention_fa3.py": "8039ffe4e9c632b41a6dba2c7b41fb25c034c3c8e820f47ad8f7b2238d118b23",
    "operators/sm90/_nosa_attention_triton.py": "c9b2060928cff8cbf594a876061546521064da78f52b3297922bcc5dd0f96985",
    "operators/sm90/_nosa_indexer_checked_cuda.py": "a59f41bdf0468a21d9e96ae180d9b8217dcbebbb07d6c7c8770700570739cdba",
    "operators/sm90/_nosa_indexer_cuda.py": "a6a518524bce510c034b8e153f7fefa05d2a3f5c0e892923328f9a85389bc35b",
    "operators/sm90/_nosa_prepare_cuda.py": "bc9c2276dabdf2c9786dd29b8b48a484df10df0ba9be8e14bc5b85f527c2bbe2",
    "operators/sm90/_nosa_prepare_ranked_cuda.py": "a2a704f1c7338400347619ca9129bc2919ba20552ffa676ca351d23762b3f237",
    "operators/sm90/_nosa_scores_cuda.py": "70a5cc50c3916675d7a23b9dca49e531a7bed9026e0f5f0bc254bd0a24bd170c",
    "operators/sm90/_nosa_selection_cuda.py": "71df015af61a1d3f554eae950e3ab57f2a641324a1fe58ae353c6e5422028a24",
    "operators/sm90/csrc/nosa_attention.cu": "01c0699b507baebe005b2df44e2f78ec149083a6239b96e7d24990edf2bf09f4",
    "operators/sm90/csrc/nosa_attention_fa3.cu": "f98b5ae4ea4bcc5c3dc5a597958627d6c186f22d1db7108162a2a328401c2f80",
    "operators/sm90/csrc/nosa_attention_grouped.cuh": "30f35b34a80c2a26c249b92e055c3158a5a2e40e5c818ff9dadc3074dc525927",
    "operators/sm90/csrc/nosa_indexer.cu": "91e4639259cd02839fa4d2606b118ff40343ac75f0419274ec4f3a868e42ed34",
    "operators/sm90/csrc/nosa_indexer_checked.cu": "49a303a3f89886ebd373611fde3faa7b6ee0adae3c8f1ebc78855734b2b1b16d",
    "operators/sm90/csrc/nosa_prepare.cu": "340f7455f11f602ac4e0380ff7bd6963e2cdaa32275bbd470ae763ea0a0a1715",
    "operators/sm90/csrc/nosa_prepare_ranked.cu": "1c40cdd8edcf6da94cc0ec59e4477035c8db91441b1c3acb76bfcdd163388d09",
    "operators/sm90/csrc/nosa_scores.cu": "d5c5b543e08a6e164701b9e5070458079c8c843f35a99de96b4c66427dfb01af",
    "operators/sm90/csrc/nosa_scores_fused.cuh": "56e6cbe675525b419db3699ae57dcf984ab64bf9e12f1bbe3b49ffd21bcd197a",
    "operators/sm90/csrc/nosa_scores_pruned.cuh": "99e0e6ce9f5685e639016e24732cce249a3b2c043ff0629b021866ad0f051409",
    "operators/sm90/csrc/nosa_selection.cu": "f2ce6b685336002d5e137ed1af05d1b2bfdda37f89b57323d053a8daa953e9e9",
    "operators/sm90/csrc/nosa_selection_cutoff.cuh": "7eaabb06b7d39d5ea0779ea209b80a3e7d54e3b137748a35f4cb47836a7f00fe",
    "operators/sm90/csrc/nosa_selection_prefix.cuh": "e5b447d6427271df99751c93e6d8c14e3cedc7fee9d76c130d8967410a3b6b35",
    "operators/sm90/nosa_attention.py": "0c1b422a5fe5c485927f017f4c2578271eaf6258d01cc3c1aa6d33d488d1d6f9",
    "operators/sm90/nosa_compression.py": "9655a9b4caeea836c846eef654f781318fc930e5ee95192fbeb6ee64dc8201f3",
    "operators/sm90/nosa_indexer.py": "9686ae49a9c63279f3f505c709bacb9b1c3c6f496880bb2508cae0377bb823da",
    "operators/sm90/nosa_validation.py": "aded63a061790919d0596c18f4129d4c77782b53c4e762ae87f7bd3aa2bb9700",
    "operators/sm90/sparse_attention.py": "5cad42677974731057cbed607d138cf8cf296dc2094c44a09b68a61f0c43a786",
    "serving/run_gr.py": "fe94fcd9011c75adbc4530759e4a8dd92d4f1a9a5f64b20b0732a5bc776a5695",
    "serving/runner.py": "bd36c4b8725bbe99f934a1ebee604a7e7b471c6c06859b3d0a1bc91c7686a421",
}
_BF16_PAIR_FORMATTED_GRAPH = _BF16_PAIR_GRAPH | {
    "operators/sm90/_nosa_attention_fa3.py": "61e2379aedaad7dddbb7237b22d014d1d37f9f251f3010c51dbe5b46b2ccde10",
}
SUPPORTED_GRAPHS += (_BF16_PAIR_GRAPH, _BF16_PAIR_FORMATTED_GRAPH)

MODEL_GROUPED_ATTENTION_SOURCE = (
    "operators/nosa/attention/device_only/csrc/nosa_attention_grouped.cuh"
)
# Current owned-resident capture: the model still executes the five GEMMs in
# [QKV, CIS, O, gate_up, down] order. The captured runner requests hidden output
# and never supplies compute_graphs or an offload cache. finite_flag therefore
# returns None: the deferred short-prefix branch is not part of this graph.
# Pin its source-level guards, migrated operator dispatch and local includes,
# plus the capture/scoping anchors; do not alias current paths to old hashes.
# Group 4 describes the native fallback; FA3 keeps its separate group-8 and
# exact launch-sequence validation below. Existing graph inventories stay intact.
_MODEL_RESIDENT_GRAPH = {
    "cache/contracts.py": "58575f1c9aee3f2843b7f2bce07f1589ea8cb9a712b050991199a95639f97f73",
    "cache/indexer_cache.py": "da80484c5d2ad016b3119f80576e72d2956f8d9d1a5cc2804a0e6d668b80b2eb",
    "cache/manager.py": "3f8a3129900f8e5e1af6005fcc187f5d430c5ce5156ca80c169384694dd39f91",
    "executor/model_executor.py": "f585f9940d5fb506e7f8aaeafe8602f9877ac773f2183603744f2b3a2a48c919",
    "experiments/indexer_block_sparse_profile/src/capture.py": "fbd4b7b877aa027d1dd2e9db9abe5672d069228b3e082285f5f05b9a783b4bae",
    "experiments/indexer_block_sparse_profile/src/instrumentation.py": "04d3ef5bd453421c65c7c83a8fd9d48b711884275456620eba220187cb35794d",
    "layers/attention.py": "91eccef789f099017c229b99d1d0ccc9d66ce980ab6dbc772a25fe031f6f34d3",
    "layers/feed_forward.py": "95dcebaa67b4f2069f580ac0831ef8335fc15b022e802708e24f58c02408de4a",
    "layers/normalization.py": "426b2cd0e1da6c944ff661a93d55eee2b064585d5c5f6c6e6a0b80ca3bc98af0",
    "models/nosa/attention.py": "62fd14e73b807bbbc5b5c3a509bdde7bb433b6a4416bd957e1e8463f9c74a0b1",
    "models/nosa/cache.py": "bc8d6898fb1d2eee7a80a689ce846656742fc6fffce2cad73ae33ebba8f62892",
    "models/nosa/config.py": "eaa168a439fada20f13b6d0a44c801de30e3b1c04d3e5ac43d4273a3454db165",
    "models/nosa/deferred_validation.py": "ca9eef34492e5c7ae7c3bf23169fecc21ca9abde3ae8c8d1670cf14fa0329ede",
    "models/nosa/indexer.py": "9202660bbcd5aba81406de0ea50ba4cbfdb2da92e7a4f65b32aab77f4fde937a",
    "models/nosa/infer.py": "5e9e8b78024d3f2073cfb858db60b95c9add264c501ecd983905cae906ea5115",
    "models/nosa/layers.py": "6f9e2a28de209df132266cdb5072f5a8581b543b1b049463338b75a2ff8ec5a9",
    "models/nosa/model.py": "febb4cafa84e36d5362c1d9b6b959860e10d1fe4fde30cd5a8c80d3b74f05335",
    "models/nosa/request_format.py": "25c775ba0b76cab91663b827fc8b38fc20511895a5e6eff3054273cbfe418825",
    "models/nosa/rotary.py": "f14c898cd77a72bd4e6908ef59b0b5fe5d42b2ebadb34aa5db092577d154dbde",
    "models/nosa/scoring.py": "c4c7e75a15a5ffb95e935b02ec482586a288c64ad079a175b293db6baa3f92d8",
    "operators/flashinfer.py": "55c368b9ffa69301b8407de7f9aa69bfce03db3b6bfc4c59646b7d3fe9d36a8d",
    "operators/nosa/_native.py": "4ed3277e55f21abac43259da7f678af74825a3ef37bbd830558bbd127f3e9b92",
    "operators/nosa/attention/common.py": "e376225202c4daf5899b870c7b30f61e571a570be6937a40456a8cf3c5d4ee69",
    "operators/nosa/attention/device_only/_cuda.py": "2f9de5fdb75e3a87ee98e46ab93ddcd18b0482b872e5f1e114238510858e5742",
    "operators/nosa/attention/device_only/_fa3.py": "64f699cf8d11d0551f70808fb733577235fda12f61b9efa8bbc4395a62158653",
    "operators/nosa/attention/device_only/_triton.py": "c9b2060928cff8cbf594a876061546521064da78f52b3297922bcc5dd0f96985",
    "operators/nosa/attention/device_only/api.py": "abbe52687cfc3d9835ea88efb31067f503f0cf07caafe95632e849f15f826f34",
    "operators/nosa/attention/device_only/csrc/nosa_attention.cu": "01c0699b507baebe005b2df44e2f78ec149083a6239b96e7d24990edf2bf09f4",
    "operators/nosa/attention/device_only/csrc/nosa_attention_fa3.cu": "f98b5ae4ea4bcc5c3dc5a597958627d6c186f22d1db7108162a2a328401c2f80",
    "operators/nosa/attention/device_only/csrc/nosa_attention_grouped.cuh": "30f35b34a80c2a26c249b92e055c3158a5a2e40e5c818ff9dadc3074dc525927",
    "operators/nosa/attention/workspace.py": "1831a2031b73d63a29359445fe2a9441712567fc82c61fdb4e1c3f0371ff5a88",
    "operators/nosa/indexer/_indexer_checked_cuda.py": "1f34a239ace16035e029123daa968e1c538d84985b8a440380dba65603337652",
    "operators/nosa/indexer/_indexer_cuda.py": "a1c80b34aef8f505468d85ceabbb48719b2a99e9beb7723091f8d1e0886629d8",
    "operators/nosa/indexer/_prepare_cuda.py": "7079fc4cbaba13d03148ba8f73b131d4ea0dca57dd81516e3c51a67b2695a5dd",
    "operators/nosa/indexer/_prepare_ranked_cuda.py": "d231b2030de13f14863eb0be1f6c5999dcd3847e7bcd771fa7399d1da18be0e1",
    "operators/nosa/indexer/_scores_cuda.py": "d5330191a95845cd702586c5596653f861074cc1dcc7e1024ead1971163fdf0a",
    "operators/nosa/indexer/_selection_cuda.py": "db6946ccc24796656cabdaf0652deb98514bc27e225ecee87046150b763245da",
    "operators/nosa/indexer/api.py": "7352cef62a22c9f980c7fb227ce8eb05458aaccfd8c6c9b1a73b0af19d0ec163",
    "operators/nosa/indexer/compression.py": "9655a9b4caeea836c846eef654f781318fc930e5ee95192fbeb6ee64dc8201f3",
    "operators/nosa/indexer/csrc/nosa_guarded_buffers.cuh": "7dadbef4c820e04e695505193ab698fc38230ac079ffc0305365a5d4d50e83fd",
    "operators/nosa/indexer/csrc/nosa_indexer.cu": "2af14714c82f10fc5bb2d6606333dbd3f6c335d982fbcf728e176307a0b9d882",
    "operators/nosa/indexer/csrc/nosa_indexer_checked.cu": "6f05e2861f724edae925581aa8971023fbf75df631fa2126fa53d5d78df8d951",
    "operators/nosa/indexer/csrc/nosa_prepare.cu": "340f7455f11f602ac4e0380ff7bd6963e2cdaa32275bbd470ae763ea0a0a1715",
    "operators/nosa/indexer/csrc/nosa_prepare_ranked.cu": "1c40cdd8edcf6da94cc0ec59e4477035c8db91441b1c3acb76bfcdd163388d09",
    "operators/nosa/indexer/csrc/nosa_scores.cu": "6b7dfa2e03f72b0e5ee43aed0cff3ff73ddd3ae85f420a57db79b0c2cdfc4bb6",
    "operators/nosa/indexer/csrc/nosa_scores_fused.cuh": "5fb278d701bef0a4cb40be56e89316e500b4c707cc4c4f36567f0986a6e97891",
    "operators/nosa/indexer/csrc/nosa_scores_pruned.cuh": "1eb136d93f04de8b252d8ddc9237f5db231faada54afa0049f223d1fec2838b8",
    "operators/nosa/indexer/csrc/nosa_selection.cu": "cc44ef81004e86e8e4da5087433195286e9924b77376e51959c8b797eb374c20",
    "operators/nosa/indexer/csrc/nosa_selection_cutoff.cuh": "7eaabb06b7d39d5ea0779ea209b80a3e7d54e3b137748a35f4cb47836a7f00fe",
    "operators/nosa/indexer/csrc/nosa_selection_prefix.cuh": "e5b447d6427271df99751c93e6d8c14e3cedc7fee9d76c130d8967410a3b6b35",
    "operators/nosa/indexer/validation.py": "c00e303f7f7c31dd6081f52da8b8225170e77f7c6f83f59a2a6995019f07ca6e",
}
SUPPORTED_GRAPHS += (_MODEL_RESIDENT_GRAPH,)

# Experiment-layout/check separation preserves the resident forward and NVTX
# scope order. The FA3 adapter change only affects the reserved-workspace entry:
# ordinary resident attention still uses the original owned-workspace dispatch.
# Keep the original graph above intact for its immutable published snapshots.
_BASELINE_RESIDENT_GRAPH = {
    name: digest
    for name, digest in _MODEL_RESIDENT_GRAPH.items()
    if not name.startswith("experiments/indexer_block_sparse_profile/")
} | {
    "experiments/nosa_baseline_performance/src/sparse/capture.py": "fbead1480d2f9956a686c02863922a060748ff3da70d7d46c2dab26fc83d8d74",
    "experiments/nosa_baseline_performance/src/sparse/instrumentation.py": "04d3ef5bd453421c65c7c83a8fd9d48b711884275456620eba220187cb35794d",
    "operators/nosa/attention/device_only/_fa3.py": "d56c29942291580bb39281858fe98908a42ee501df460ca700a7a1bcb7949254",
}
SUPPORTED_GRAPHS += (_BASELINE_RESIDENT_GRAPH,)


def _validate_source_graph(hashes):
    """Validate attribution anchors and return the reviewed attention group size."""
    for graph in reversed(SUPPORTED_GRAPHS):
        if all(hashes.get(name) == expected for name, expected in graph.items()):
            return (
                4
                if GROUPED_ATTENTION_SOURCE in graph or MODEL_GROUPED_ATTENTION_SOURCE in graph
                else 1
            )
    raise ValueError("Unreviewed inference graph for unscoped GEMM attribution")


def _cached_indexer(workload):
    execution = workload.get("indexer_execution")
    revisions = NATIVE_INDEXER_REVISIONS
    if execution not in (None, "cached_flashinfer_v1", *revisions):
        raise ValueError(f"Unsupported indexer_execution: {execution}")
    cached = execution is not None
    if execution in revisions and (
        workload.get("native_kernel_revision") != revisions[execution]
        or workload.get("selection_backend") != "cuda_tvm_ffi"
        or workload.get("kernel_backend") != "cuda_tvm_ffi"
    ):
        raise ValueError(f"{execution} requires the reviewed native kernel revision and selection")
    if cached and workload.get("indexer_query_chunk_size") is not None:
        raise ValueError(f"{execution} requires whole-batch query dispatch")
    return cached


def _matrix_kernel(kernel):
    name = kernel["name"].lower()
    return name.startswith("nvjet_") or "gemm" in name


def _kernel_family(kernel, family):
    return re.search(rf"(?<![\w:]){re.escape(family)}(?=[<(\s]|$)", kernel["name"]) is not None


def _fused_selection_flag(kernel, family=FUSED_SCORE_FAMILY):
    """Read the Selection template parameter, not the misleading score family name.

    Revision 3 instantiates <T, O, Pool, Selection, Map>. Nested template commas
    in dtype/Map names must not change which argument is interpreted as a bool.
    Older score-only captures may lack the Selection parameter altogether.
    """
    match = re.search(re.escape(family) + r"\s*<", kernel["name"])
    if match is None:
        return None
    arguments, current, depth = [], [], 0
    for character in kernel["name"][match.end() :]:
        if character == "<":
            depth += 1
        elif character == ">":
            if depth == 0:
                arguments.append("".join(current).strip())
                break
            depth -= 1
        if character == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
            if len(arguments) == 4:
                break
        else:
            current.append(character)
    booleans = {"true": True, "false": False, "1": True, "0": False}
    arguments = [re.sub(r"^\(bool\)\s*", "", argument) for argument in arguments]
    if len(arguments) < 4 or arguments[2] not in booleans:
        return None
    return booleans.get(arguments[3])


def _shared_selection(query_length, total_length):
    start = total_length - query_length
    first = start // 64
    return (
        query_length >= 128
        and (total_length + 63) // 64 <= 1056
        and first >= 64
        and (total_length - 1) // 64 - first <= 16
    )


def _joint_submission(query_length, total_length, revision):
    return (
        query_length >= (128 if revision >= 3 else 1024)
        and total_length // 16 - 1 >= 2047
        and (total_length + 63) // 64 <= 1056
    )


def _native_selection_kernels(query_length, total_length, *, prepared_ranking=False):
    """Exact BF16 whole-batch dispatch; ranked preparation removes one launch."""
    shared = _shared_selection(query_length, total_length)
    if prepared_ranking and not shared:
        raise ValueError("Prepared ranking requires the reviewed causal prefix selection path")
    if not shared:
        return ("nosa_selection::selection_kernel",)
    return (() if prepared_ranking else ("nosa_selection::prepare_prefix_ranking",)) + (
        "nosa_selection::selection_prefix_kernel",
    )


def _validate_operator_kernels(
    stage,
    selected,
    backend,
    *,
    query_length,
    total_length,
    native_attention_group_size=4,
    native_revision=1,
    prepared_ranking=False,
    num_kv_heads=2,
    attention_execution=None,
):
    """Require the reviewed call's complete kernel sequence and fusion kind."""
    if native_revision not in (1, 2, 3, 4, 5):
        raise ValueError("Unreviewed native kernel revision")
    if stage == "native_checked_indexer":
        if (
            native_revision < 4
            or not prepared_ranking
            or not _joint_submission(query_length, total_length, native_revision)
        ):
            raise ValueError("Checked native scope requires the reviewed ranked joint dispatch")
        options = {
            "query_length": query_length,
            "total_length": total_length,
            "native_attention_group_size": native_attention_group_size,
            "native_revision": native_revision,
            "prepared_ranking": True,
            "num_kv_heads": num_kv_heads,
        }
        _validate_operator_kernels("native_prepare_ranked", selected[:2], backend, **options)
        return _validate_operator_kernels("native_indexer", selected[2:], backend, **options)
    score_backend = None
    alternatives = []
    selection_fused = False
    if stage == "block_sparse_attention":
        if backend == "cuda_tvm_ffi":
            expected = ("nosa_attention::attention_kernel",)
            if native_attention_group_size == 4 and query_length >= 4:
                expected = ("nosa_attention::grouped_attention_kernel", *expected)
                if native_revision >= 2:
                    expected = ("nosa_attention::nonfinite_blocks_kernel", *expected)
            if attention_execution in ("native_fa3_v2", "native_fa3_v3"):
                if query_length >= 1:
                    alternatives.append(expected)
                    expected = FA3_ATTENTION_SEQUENCE
                    if attention_execution == "native_fa3_v3":
                        heads = _integer(num_kv_heads, "num_kv_heads", minimum=1)
                        rows = _integer(query_length, "query_length", minimum=1)
                        if (rows + 7) // 8 * heads == 256:
                            expected = FA3_SORTED_ATTENTION_SEQUENCE
                elif not query_length:
                    expected = ()
        else:
            expected = ("_nosa_block_attention",)
    elif stage in ("compressed_scores", "pooled_scores", "native_indexer"):
        compressed_count = max(0, (total_length - 32) // 16 + 1)
        native_min_count = 2047 if query_length >= 1024 else 511
        score_backend = (
            "cuda_tvm_ffi"
            if backend == "cuda_tvm_ffi" and compressed_count >= native_min_count
            else "triton"
        )
        expected = (
            ("nosa_scores::normalizer_kernel", "nosa_scores::scores_kernel")
            if score_backend == "cuda_tvm_ffi"
            else ("_scores",)
        )
        score_fused = (
            native_revision >= 2
            and score_backend == "cuda_tvm_ffi"
            and stage != "compressed_scores"
            and query_length >= 1024
            and (query_length + 15) // 16 * num_kv_heads >= 128
        )
        if score_fused:
            if native_revision >= 3:
                expected = (FUSED_SCORE_FAMILY,)
            else:
                alternatives.append((FUSED_SCORE_FAMILY,))
        if stage == "native_indexer":
            if native_revision < 2 or backend != "cuda_tvm_ffi":
                raise ValueError("Native joint submission requires the reviewed native revision")
            if not _joint_submission(query_length, total_length, native_revision):
                raise ValueError("Native joint scope disagrees with its query geometry")
            selection_fused = (
                native_revision >= 3
                and score_fused
                and query_length <= 1088
                and _shared_selection(query_length, total_length)
            )
            if selection_fused:
                expected = (
                    () if prepared_ranking else ("nosa_selection::prepare_prefix_ranking",)
                ) + (FUSED_SCORE_FAMILY,)
                alternatives = []
                if (
                    native_revision >= 5
                    and query_length == 1024
                    and total_length == 66560
                    and num_kv_heads == 2
                ):
                    # A failed resource guard retains the original fused kernel.
                    alternatives.append(expected)
                    expected = (*expected[:-1], PRUNED_SCORE_FAMILY)
            else:
                suffix = _native_selection_kernels(
                    query_length, total_length, prepared_ranking=prepared_ranking
                )
                expected += suffix
                alternatives = [sequence + suffix for sequence in alternatives]
    elif stage == "native_selection":
        if native_revision < 2 or backend != "cuda_tvm_ffi":
            raise ValueError("Native selection requires the reviewed native revision")
        expected = _native_selection_kernels(
            query_length, total_length, prepared_ranking=prepared_ranking
        )
    elif stage in ("native_prepare", "native_prepare_ranked"):
        if native_revision < 2 or backend != "cuda_tvm_ffi":
            raise ValueError("Native preparation requires the reviewed native revision")
        if stage == "native_prepare_ranked":
            if native_revision < 3 or not _shared_selection(query_length, total_length):
                raise ValueError("Ranked preparation disagrees with the reviewed append geometry")
            expected = (
                "nosa_prepare::finite_partials",
                "nosa_prepare_ranked::guarded_compression_and_ranking",
            )
        else:
            expected = ("nosa_prepare::finite_partials", "nosa_prepare::guarded_compression")
    else:
        return None

    def matches(sequence):
        if len(selected) != len(sequence):
            return False
        for family, kernel in zip(sequence, selected, strict=True):
            if not _kernel_family(kernel, family):
                return False
            if family == "flashinfer::PrefillWithKVCacheKernel" and not FA3_MAIN_TEMPLATE.search(
                kernel["name"]
            ):
                return False
            if family in FUSED_SCORE_FAMILIES:
                flag = _fused_selection_flag(kernel, family)
                if flag is not selection_fused and not (native_revision < 3 and flag is None):
                    return False
        return True

    if not any(matches(sequence) for sequence in (expected, *alternatives)):
        raise ValueError(
            f"Scope {stage} must contain exactly a reviewed kernel sequence: {(expected, *alternatives)}"
        )
    return score_backend


def _interval(record, label, *, start="start"):
    left = _integer(record.get(start), f"{label}.{start}")
    right = _integer(record.get("end"), f"{label}.end")
    if right <= left:
        raise ValueError(f"{label} interval must have positive duration")
    return left, right


def _workload_calls(phase, num_layers, workload):
    prefix = _integer(workload["prefix_tokens"], "prefix_tokens")
    new = _integer(workload["new_tokens"], "new_tokens", minimum=1)
    total = _integer(workload.get("total_tokens", prefix + new), "total_tokens", minimum=1)
    chunk = _integer(workload["chunk_size"], "chunk_size", minimum=1)
    if total != prefix + new:
        raise ValueError("total_tokens must equal prefix_tokens + new_tokens")
    first = 0 if phase == "full_prefill" else prefix
    return [
        (layer, start, min(chunk, total - start))
        for start in range(first, total, chunk)
        for layer in range(num_layers)
    ]


def _empty_module(name):
    return {
        "kernel_ns": 0,
        "kernel_ms": 0.0,
        "kernel_count": 0,
        "parent_module": PARENTS.get(name),
        "inclusive": name in INCLUSIVE_MODULES,
    }


def _add_kernels(target, kernels):
    target["kernel_ns"] += sum(k["end"] - k["start"] for k in kernels)
    target["kernel_ms"] = target["kernel_ns"] / 1e6
    target["kernel_count"] += len(kernels)


def _scope_kernels(scope, kernels, launches):
    return kernels[bisect_left(launches, scope["start"]) : bisect_left(launches, scope["end"])]


def _require_marker(kernels, launches, begin, end, marker):
    selected = kernels[bisect_right(launches, begin) : bisect_left(launches, end)]
    matches = [k for k in selected if marker.lower() in k["name"].lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {marker} kernel between its GEMM anchors")


def _attribute_run(
    root, scopes, kernels, num_layers, workload, native_attention_group_size, num_kv_heads
):
    kernels = sorted(kernels, key=lambda k: k["launch_start"])
    launches = [k["launch_start"] for k in kernels]
    calls = _workload_calls(root["phase"], num_layers, workload)
    grouped = defaultdict(lambda: defaultdict(list))
    for scope in scopes:
        key = (scope["layer_idx"], scope["query_start"], scope["query_length"])
        grouped[key][scope["stage"]].append(scope)
    if set(grouped) != set(calls):
        raise ValueError("NVTX layer/query calls do not match the complete workload")
    primary = sorted(
        (scope for scope in scopes if scope["stage"] in PRIMARY_STAGES),
        key=lambda scope: scope["start"],
    )
    if any(left["end"] > right["start"] for left, right in pairwise(primary)):
        raise ValueError("Primary layer scopes must not overlap")
    primary_starts = [scope["start"] for scope in primary]

    def primary_owner(kernel):
        position = bisect_right(primary_starts, kernel["launch_start"]) - 1
        if position >= 0 and kernel["launch_start"] < primary[position]["end"]:
            return primary[position]
        return None

    # QK/PV kernels may have "gemm" in a demangled template name. Their
    # enclosing scope owns that work; only CIS and unscoped model GEMMs enter
    # the reviewed five-projection sequence.
    matrices = [
        kernel
        for kernel in kernels
        if _matrix_kernel(kernel)
        and ((owner := primary_owner(kernel)) is None or owner["stage"] == "cis_projection")
    ]
    if len(matrices) != 5 * len(calls):
        raise ValueError("Expected five GEMM kernels per layer/query call, four outside CIS")
    modules = {name: _empty_module(name) for name in MODULES}
    covered = set()
    layer_calls = []
    score_dispatch = Counter()
    attention_dispatch = Counter()
    cached = _cached_indexer(workload)
    backend = validate_kernel_backend(workload)
    query_tile = (
        None
        if cached
        else _integer(
            workload.get("indexer_query_chunk_size", 64), "indexer_query_chunk_size", minimum=1
        )
    )
    for call_index, key in enumerate(calls):
        group = grouped[key]
        if any(len(group[stage]) != 1 for stage in PRIMARY_STAGES):
            raise ValueError("Each layer call needs exactly one scope for each primary stage")
        cis, indexer, attention = [group[name][0] for name in PRIMARY_STAGES]
        if not (cis["end"] <= indexer["start"] and indexer["end"] <= attention["start"]):
            raise ValueError("Expected CIS -> indexer -> block attention scope order")
        nested = sorted(
            (scope for stage in NESTED_STAGES for scope in group[stage]), key=lambda s: s["start"]
        )
        for scope in nested:
            if not (indexer["start"] <= scope["start"] < scope["end"] <= indexer["end"]):
                raise ValueError("Indexer child scope is not nested in its matching parent")
        if any(left["end"] > right["start"] for left, right in pairwise(nested)):
            raise ValueError("Indexer child scopes must not overlap")
        layer, start, length = key
        scored = (start + length + 63) // 64 > 64
        if cached:
            expected = list(FUSED_STAGES if scored else FUSED_STAGES[:2])
            if scored and workload.get("indexer_execution") in (
                "cached_native_v2",
                "cached_native_v3",
                "cached_native_v4",
                "cached_native_v5",
            ):
                joint = _joint_submission(
                    length, start + length, workload.get("native_kernel_revision", 1)
                )
                expected = [
                    *FUSED_STAGES[:2],
                    *(["native_indexer"] if joint else ["pooled_scores", "native_selection"]),
                ]
            if workload.get("indexer_preparation") == "native_guarded_v1":
                expected[:2] = ["native_prepare"]
            elif workload.get("indexer_preparation") in (
                "native_guarded_ranked_v1",
                "native_guarded_ranked_checked_v1",
            ):
                expected[:2] = [
                    "native_prepare_ranked"
                    if _shared_selection(length, start + length)
                    else "native_prepare"
                ]
                if (
                    workload.get("indexer_preparation") == "native_guarded_ranked_checked_v1"
                    and _shared_selection(length, start + length)
                    and _joint_submission(length, start + length, 4)
                ):
                    expected = ["native_checked_indexer"]
        else:
            tiles = (length + query_tile - 1) // query_tile if scored else 0
            expected = ["compression_k", "compression_cis"] + [
                stage for _ in range(tiles) for stage in ("compressed_scores", "select_from_scores")
            ]
        if [scope["stage"] for scope in nested] != expected:
            raise ValueError(
                "Indexer child count/order disagrees with query tiling or short-context bypass"
            )
        per_call = {name: _empty_module(name) for name in MODULES if name != "other_non_matrix"}
        prepared_ranking = bool(group["native_prepare_ranked"] or group["native_checked_indexer"])
        for stage in STAGES:
            for scope_index, scope in enumerate(group[stage]):
                selected = _scope_kernels(scope, kernels, launches)
                if not selected and not (cached and stage == "indexer_cache_update"):
                    raise ValueError(f"Scope {stage} contains no correlated CUDA kernels")
                score_rows = (
                    min(query_tile, length - scope_index * query_tile)
                    if stage == "compressed_scores"
                    else length
                )
                score_backend = _validate_operator_kernels(
                    stage,
                    selected,
                    backend,
                    query_length=score_rows,
                    total_length=start + length,
                    native_attention_group_size=native_attention_group_size,
                    native_revision=workload.get("native_kernel_revision", 1),
                    prepared_ranking=prepared_ranking,
                    num_kv_heads=num_kv_heads,
                    attention_execution=workload.get("attention_execution"),
                )
                if stage == "block_sparse_attention":
                    actual_attention = (
                        workload["attention_execution"]
                        if any(_kernel_family(k, "nosa_fa3::prepare") for k in selected)
                        else "native_grouped_v2"
                        if any(
                            _kernel_family(k, "nosa_attention::grouped_attention_kernel")
                            for k in selected
                        )
                        else "native_per_query_v1"
                        if backend == "cuda_tvm_ffi"
                        else "triton_v1"
                    )
                    attention_dispatch[actual_attention] += 1
                if score_backend is not None:
                    score_dispatch[score_backend] += 1
                if stage in ("native_indexer", "native_checked_indexer"):
                    preparation = (
                        (
                            ("native_finite_check", selected[:1]),
                            ("native_ranked_compression", selected[1:2]),
                        )
                        if stage == "native_checked_indexer"
                        else ()
                    )
                    if preparation:
                        selected = selected[2:]
                    fused = [
                        kernel
                        for kernel in selected
                        if any(
                            _kernel_family(kernel, family)
                            and _fused_selection_flag(kernel, family) is True
                            for family in FUSED_SCORE_FAMILIES
                        )
                    ]
                    scores = [
                        kernel
                        for kernel in selected
                        if "nosa_scores::" in kernel["name"] and kernel not in fused
                    ]
                    selection = [
                        kernel for kernel in selected if "nosa_selection::" in kernel["name"]
                    ]
                    if len(scores) + len(selection) + len(fused) != len(selected):
                        raise ValueError("Unattributed kernel within native joint submission")
                    for name, family in (
                        ("pooled_scores", scores),
                        ("native_selection", selection),
                        ("score_selection", fused),
                        *preparation,
                    ):
                        _add_kernels(modules[name], family)
                        _add_kernels(per_call[name], family)
                else:
                    _add_kernels(modules[stage], selected)
                    _add_kernels(per_call[stage], selected)
                if stage in PRIMARY_STAGES:
                    covered.update(k["correlationId"] for k in selected)
        qk_modules = [name for name in QK_MODULES if per_call[name]["kernel_count"]]
        if len(qk_modules) != int(scored):
            raise ValueError(
                "Each scored call must own QK in exactly one score or score+selection module"
            )
        gemms = dict(zip(LINEARS, matrices[5 * call_index : 5 * (call_index + 1)], strict=True))
        for name, kernel in gemms.items():
            owner = primary_owner(kernel)
            if name == "cis_projection":
                if owner is not cis:
                    raise ValueError(
                        "Five-GEMM sequence has a CIS projection outside its matching CIS scope"
                    )
            elif owner is not None:
                raise ValueError("The four non-CIS GEMMs must be outside all primary scopes")
        if not (gemms["qkv_proj"]["launch_start"] < cis["start"]):
            raise ValueError("QKV GEMM must precede its CIS scope")
        if gemms["o_proj"]["launch_start"] < attention["end"]:
            raise ValueError("Output projection GEMM must follow its block attention scope")
        _require_marker(
            kernels,
            launches,
            gemms["qkv_proj"]["launch_start"],
            gemms["cis_projection"]["launch_start"],
            "BatchQKApplyRotary",
        )
        _require_marker(
            kernels,
            launches,
            gemms["o_proj"]["launch_start"],
            gemms["gate_up_proj"]["launch_start"],
            "fused_add_rmsnorm",
        )
        _require_marker(
            kernels,
            launches,
            gemms["gate_up_proj"]["launch_start"],
            gemms["down_proj"]["launch_start"],
            "act_and_mul",
        )
        for name, kernel in gemms.items():
            module = "cis_projection_gemm" if name == "cis_projection" else name
            _add_kernels(modules[module], [kernel])
            _add_kernels(per_call[module], [kernel])
            covered.add(kernel["correlationId"])
        layer_calls.append(
            {
                "layer_idx": layer,
                "query_start": start,
                "query_length": length,
                "indexer_qk_module": qk_modules[0] if qk_modules else None,
                "native_joint_submission": bool(
                    group["native_indexer"] or group["native_checked_indexer"]
                ),
                "prepared_ranking": prepared_ranking,
                "attention_execution": actual_attention,
                "modules": per_call,
                "gemm_correlations": {
                    name: kernel["correlationId"] for name, kernel in gemms.items()
                },
                "cis_scope": {name: cis[name] for name in ("start", "end")},
                "attention_scope": {name: attention[name] for name in ("start", "end")},
            }
        )
    other = [kernel for kernel in kernels if kernel["correlationId"] not in covered]
    if any(_matrix_kernel(kernel) for kernel in other):
        raise ValueError("Unattributed matrix kernel remains after layer sequence reconstruction")
    _add_kernels(modules["other_non_matrix"], other)
    total_ns = sum(kernel["end"] - kernel["start"] for kernel in kernels)
    partition = [row for name, row in modules.items() if name not in PARENTS]
    if sum(row["kernel_ns"] for row in partition) != total_ns or sum(
        row["kernel_count"] for row in partition
    ) != len(kernels):
        raise ValueError(
            "Exclusive module partition must conserve all correlated kernel time/counts"
        )
    return {
        "phase": root["phase"],
        "iteration": root["iteration"],
        "kernel_ms": total_ns / 1e6,
        "kernel_count": len(kernels),
        "score_dispatch_scope_counts": dict(sorted(score_dispatch.items())),
        "attention_dispatch_scope_counts": dict(sorted(attention_dispatch.items())),
        "modules": modules,
        "layer_calls": layer_calls,
        "root_scope": {name: root[name] for name in ("start", "end", "text", "globalTid")},
    }


def attribute_kernels(
    scopes, kernels, num_layers, workload, *, native_attention_group_size=4, num_kv_heads=2
):
    """Attribute launch-correlated kernels, independent of SQLite or model FLOPs.

    Times are integer nanoseconds. Scopes need start/end/text/globalTid;
    kernels need start/end/launch_start/globalTid/correlationId/name/deviceId/
    streamId. Optional globalPid/contextId are also checked for uniqueness.
    GPU start/end may lie outside the CPU NVTX interval: launch_start owns the
    attribution, while end-start supplies the measured denominator.

    Returns per-phase/per-iteration module dictionaries plus per-layer calls.
    The fixed source graph is an external precondition, verified by analyze().
    Its revision supplies native_attention_group_size; the default describes
    the preserved four-query native fallback, while old snapshots use 1.
    FA3 has an independent eight-query mainloop and three or four launches.
    Synthetic workloads are supported here to test attribution independently.
    """
    _integer(num_layers, "num_layers", minimum=1)
    _integer(num_kv_heads, "num_kv_heads", minimum=1)
    if native_attention_group_size not in (1, 4):
        raise ValueError("Unsupported reviewed native attention group size")
    roots, stages = [], []
    for source in scopes:
        text = source.get("text", "")
        if not text.startswith("NOSA/"):
            continue
        scope = dict(source)
        _interval(scope, "NVTX scope")
        _integer(scope.get("globalTid"), "scope.globalTid")
        if match := ROOT_RANGE.fullmatch(text):
            scope.update(phase=match[1], iteration=int(match[2]))
            roots.append(scope)
        elif match := STAGE_RANGE.fullmatch(text):
            if match[3] not in STAGES:
                raise ValueError(f"Unknown NOSA stage: {match[3]}")
            scope.update(
                phase=match[1],
                layer_idx=int(match[2]),
                stage=match[3],
                query_start=int(match[4]),
                query_length=int(match[5]),
            )
            stages.append(scope)
        else:
            raise ValueError(f"Unrecognized NOSA NVTX range: {text}")
    roots.sort(key=lambda root: root["start"])
    if not roots or any(a["end"] > b["start"] for a, b in pairwise(roots)):
        raise ValueError("Expected nonoverlapping profile root ranges")
    counts = Counter(root["phase"] for root in roots)
    if set(counts) != set(PHASES) or len(set(counts.values())) != 1:
        raise ValueError("Both profile phases must contain the same complete run count")
    repeats = workload.get("profile_repeats", counts["extend"])
    _integer(repeats, "profile_repeats", minimum=1)
    for phase in PHASES:
        if sorted(root["iteration"] for root in roots if root["phase"] == phase) != list(
            range(repeats)
        ):
            raise ValueError("Profile root iterations must exactly cover profile_repeats")
    threads = {scope["globalTid"] for scope in (*roots, *stages)}
    if len(threads) != 1:
        raise ValueError("Module reconstruction requires exactly one CPU launch thread")
    kernels = [dict(kernel) for kernel in kernels]
    if not kernels:
        raise ValueError("No correlated CUDA kernels were supplied")
    for kernel in kernels:
        _interval(kernel, "kernel")
        for field in ("launch_start", "globalTid", "correlationId", "deviceId", "streamId"):
            _integer(kernel.get(field), f"kernel.{field}")
        if kernel["launch_start"] > kernel["start"]:
            raise ValueError("CUDA kernel begins before its correlated host launch")
        if not isinstance(kernel.get("name"), str) or not kernel["name"]:
            raise ValueError("CUDA kernel names are required")
    if {kernel["globalTid"] for kernel in kernels} != threads:
        raise ValueError("Kernels must originate from the profile's single launch thread")
    for field in ("deviceId", "streamId", "globalPid", "contextId"):
        values = {kernel.get(field) for kernel in kernels}
        if len(values) != 1:
            raise ValueError(f"Module reconstruction requires exactly one {field}")
    if len({kernel["correlationId"] for kernel in kernels}) != len(kernels):
        raise ValueError("Every kernel requires a unique launch correlationId")
    launch_order = sorted(kernels, key=lambda kernel: kernel["launch_start"])
    if any(
        a["launch_start"] == b["launch_start"] or a["start"] > b["start"]
        for a, b in pairwise(launch_order)
    ):
        raise ValueError("Single-stream GPU order must match the unambiguous host launch order")
    overlaps = []
    for left, right in pairwise(launch_order):
        overlap = max(0, left["end"] - right["start"])
        if overlap:
            if overlap > 1000:
                raise ValueError(
                    "GPU kernel overlap exceeds the small nsys timestamp-boundary tolerance"
                )
            overlaps.append(overlap)
    starts = [root["start"] for root in roots]
    root_scopes, root_kernels = defaultdict(list), defaultdict(list)
    for scope in stages:
        index = bisect_right(starts, scope["start"]) - 1
        if (
            index < 0
            or scope["end"] > roots[index]["end"]
            or scope["phase"] != roots[index]["phase"]
        ):
            raise ValueError("Every stage scope must belong to its matching profile root")
        root_scopes[index].append(scope)
    excluded = []
    for kernel in kernels:
        index = bisect_right(starts, kernel["launch_start"]) - 1
        if index >= 0 and kernel["launch_start"] < roots[index]["end"]:
            root_kernels[index].append(kernel)
        else:
            excluded.append(kernel)
    result = {
        "runs": [
            _attribute_run(
                root,
                root_scopes[index],
                root_kernels[index],
                num_layers,
                workload,
                native_attention_group_size,
                num_kv_heads,
            )
            for index, root in enumerate(roots)
        ],
        "validation": {
            "single_launch_thread": next(iter(threads)),
            "device_id": kernels[0]["deviceId"],
            "stream_id": kernels[0]["streamId"],
            "global_pid": kernels[0].get("globalPid"),
            "context_id": kernels[0].get("contextId"),
            "correlation_count": len(kernels),
            "excluded_outside_profile_roots": len(excluded),
            "overlap_count": len(overlaps),
            "overlap_ns": sum(overlaps),
            "max_overlap_ns": max(overlaps, default=0),
            "overlap_tolerance": "<=1000 ns on the verified single CUDA stream",
            "gemm_order": list(LINEARS),
            "native_attention_group_size": native_attention_group_size,
            "fa3_attention_group_size": 8
            if workload.get("attention_execution") in ("native_fa3_v2", "native_fa3_v3")
            else None,
            "num_key_value_heads": num_kv_heads,
            "time_definition": "Sum of kernel end-start; not union active time or CUDA-event span",
        },
    }
    return result


def read_trace(path):
    """Read an exported SQLite trace without writing indexes or modifying it."""
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        scopes = [
            dict(row)
            for row in connection.execute(
                "SELECT n.start,n.end,COALESCE(n.text,s.value) AS text,n.globalTid "
                "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
                "WHERE COALESCE(n.text,s.value) LIKE 'NOSA/%'"
            )
        ]
        kernels = [
            dict(row)
            for row in connection.execute(
                "SELECT k.start,k.end,r.start AS launch_start,r.globalTid,k.correlationId,"
                "s.value AS name,k.deviceId,k.streamId,k.contextId,k.globalPid "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId "
                "LEFT JOIN StringIds s ON k.demangledName=s.id ORDER BY r.start"
            )
        ]
    return scopes, kernels


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_trace_gpu(path, device_id, gpu, *, global_pid=None):
    """Bind a process-local CUDA ordinal to the captured physical GPU.

    CUDA_VISIBLE_DEVICES can make a kernel's deviceId differ from
    TARGET_INFO_GPU.id. Modern nsys exports record the process-local mapping
    in TARGET_INFO_CUDA_DEVICE; PROCESSES binds it to the kernel globalPid.
    Never infer that mapping from a matching model name or UUID alone.
    """
    _integer(device_id, "kernel.deviceId")

    def uuid(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("GPU UUID is required in SQLite and metadata")
        return value.lower().removeprefix("gpu-")

    mapping = None
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        physical_id = device_id
        if "TARGET_INFO_CUDA_DEVICE" in tables:
            _integer(global_pid, "kernel.globalPid")
            if "PROCESSES" not in tables:
                raise ValueError("SQLite CUDA device mapping requires the kernel process table")
            processes = connection.execute(
                "SELECT pid FROM PROCESSES WHERE globalPid=?", (global_pid,)
            ).fetchall()
            if len(processes) != 1:
                raise ValueError("SQLite must identify exactly one process for kernel globalPid")
            pid = _integer(processes[0]["pid"], "process.pid")
            mappings = connection.execute(
                "SELECT gpuId,cudaId,pid,uuid,numMultiprocessors "
                "FROM TARGET_INFO_CUDA_DEVICE WHERE pid=? AND cudaId=?",
                (pid, device_id),
            ).fetchall()
            if len(mappings) != 1:
                raise ValueError(
                    "SQLite must identify exactly one process-local CUDA device mapping"
                )
            mapping = dict(mappings[0])
            physical_id = _integer(mapping["gpuId"], "CUDA device mapping.gpuId")
        rows = connection.execute(
            "SELECT * FROM TARGET_INFO_GPU WHERE id=?", (physical_id,)
        ).fetchall()
    if len(rows) != 1:
        raise ValueError("SQLite must identify exactly one GPU for the kernel device ID")
    device = dict(rows[0])
    if uuid(device["uuid"]) != uuid(gpu.get("uuid")):
        raise ValueError("SQLite GPU UUID differs from benchmark metadata")
    if device["smCount"] != gpu.get("sm_count") or [
        device["computeMajor"],
        device["computeMinor"],
    ] != gpu.get("capability"):
        raise ValueError("SQLite GPU architecture differs from benchmark metadata")
    if "total_memory" in gpu and device.get("totalMemory") != gpu["total_memory"]:
        raise ValueError("SQLite GPU memory differs from benchmark metadata")
    if mapping is not None:
        if mapping["uuid"] is not None and uuid(mapping["uuid"]) != uuid(device["uuid"]):
            raise ValueError("SQLite CUDA device mapping UUID differs from its physical GPU")
        if mapping["numMultiprocessors"] != device["smCount"]:
            raise ValueError("SQLite CUDA device mapping SM count differs from its physical GPU")
        device["cuda_device_mapping"] = mapping | {"globalPid": global_pid}
    device["kernel_device_id"] = device_id
    return device


def _validate_inputs(data_dir, metadata, summary, profile_metadata):
    if profile_metadata.get("run_id") != metadata.get("run_id"):
        raise ValueError("Profile and benchmark run_id must agree")
    if profile_metadata.get("args", {}).get("mode") != "profile":
        raise ValueError("Expected profile-mode metadata")
    for key in (
        "model_config",
        "source_sha256",
        "request_sha256",
        "checkpoint_config_sha256",
        "gpu",
        "torch",
        "cuda",
        "triton",
        "flashinfer",
        "tvm_ffi",
        "native_build",
    ):
        if profile_metadata.get(key) != metadata.get(key):
            raise ValueError(f"Profile and benchmark metadata disagree on {key}")
    validate_kernel_backend(summary["workload"], metadata)
    validate_kernel_backend(summary["workload"], profile_metadata)
    for key in ("prefix_tokens", "new_tokens", "chunk_size", "profile_repeats", "device"):
        if profile_metadata["args"][key] != metadata["args"][key]:
            raise ValueError(f"Profile and benchmark arguments disagree on {key}")
    if "numerical_acceptance" in metadata or "numerical_acceptance" in profile_metadata:
        from experiments.nosa_baseline_performance.src.acceptance import verify_acceptance

        benchmark_check = verify_acceptance(data_dir, metadata, "sparse")
        profile_check = verify_acceptance(data_dir, profile_metadata, "sparse")
        if benchmark_check["receipt_sha256"] != profile_check["receipt_sha256"]:
            raise ValueError("Benchmark and profile acceptance receipts differ")
        evidence = Path(profile_check["artifact_paths"]["attention_audit"])
        if _sha256(evidence) != _sha256(data_dir / "attention_audit.json"):
            raise ValueError("Geometry audit differs from independently accepted evidence")
    elif profile_metadata.get("validation", {}).get("finite") is not True:
        raise ValueError("The captured profile must have passed its finite-output check")
    hashes = metadata["source_sha256"]
    _validate_source_graph(hashes)
    verified = {}
    for name, expected in hashes.items():
        path = (data_dir / "sources" / name).resolve()
        if not path.is_relative_to((data_dir / "sources").resolve()) or _sha256(path) != expected:
            raise ValueError(f"Captured source snapshot hash mismatch: {name}")
        verified[name] = expected
    config = metadata["model_config"]
    if config.get("attention_bias") or config.get("mlp_bias") or config.get("tie_word_embeddings"):
        raise ValueError("Only the reviewed bias-free NOSA inference graph is supported")
    # Shapes come from the captured model configuration and the independent
    # untimed resident-geometry audit, never from a GEMM kernel's name/grid.
    audit = json.loads((data_dir / "attention_audit.json").read_text())
    calls = [record for record in audit if record["stage"] == "block_sparse_attention"]
    if sorted(record["layer_idx"] for record in calls) != list(range(config["num_hidden_layers"])):
        raise ValueError("Geometry audit must cover every layer exactly once")
    workload = summary["workload"]
    heads, kv_heads, dim = (
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
    )
    for record in calls:
        details = record["details"]
        expected = {
            "q_shape": [workload["new_tokens"], heads, dim],
            "k_shape": [workload["total_tokens"], kv_heads, dim],
            "v_shape": [workload["total_tokens"], kv_heads, dim],
            "selection_shape": [workload["new_tokens"], kv_heads, 64],
            "block_size": 64,
            "block_budget": 64,
            "valid_blocks_min": 64,
            "valid_blocks_max": 64,
        }
        if record.get("status") != "ok" or record.get("phase") != "extend":
            raise ValueError("Geometry audit must contain successful extend calls")
        if (
            record["query_start"] != workload["prefix_tokens"]
            or record["query_length"] != workload["new_tokens"]
        ):
            raise ValueError("Geometry audit query range differs from the workload")
        if any(details.get(name) != value for name, value in expected.items()):
            raise ValueError(
                "Resident Q/K/V/selection shape or block budget audit disagrees with config"
            )
    return verified


def _flops_by_module(config, prefix, query, chunk_size, *, cached=False, qk_module=None):
    flops = matrix_flops(config, prefix, query, chunk_size)
    if qk_module is None:
        qk_module = "pooled_scores" if cached else "compressed_scores"
    if qk_module not in QK_MODULES:
        raise ValueError("Useful indexer QK requires a score or score+selection module")
    return {
        **{name: flops.get(name, 0) for name in MODULES},
        "indexer_total": flops["indexer_qk"],
        **{name: flops["indexer_qk"] if name == qk_module else 0 for name in QK_MODULES},
        "cis_projection_gemm": flops["cis_projection"],
    }


def _add_mfu(module, flops, peak):
    module["matrix_flops"] = flops
    if flops:
        if module["kernel_ms"] <= 0:
            raise ValueError("Positive matrix work requires a positive measured kernel duration")
        module["effective_tflops"] = flops / module["kernel_ms"] / 1e9
        module["mfu_pct"] = module["effective_tflops"] / peak * 100
    else:
        module["effective_tflops"] = module["mfu_pct"] = None


def build_module_report(metadata, summary, profile_metadata, attribution, *, peak_tflops=None):
    """Attach useful matrix FLOPs to each attributed run and preserve E2E MFU."""
    baseline = build_report(metadata, summary, peak_tflops=peak_tflops)
    peak = baseline["peak_tflops"]
    config, workload = metadata["model_config"], summary["workload"]
    cached = _cached_indexer(workload)
    runs = attribution["runs"]
    repeats = _integer(workload["profile_repeats"], "profile_repeats", minimum=1)
    for phase in PHASES:
        if (
            sum(run["phase"] == phase for run in runs) != repeats
            or summary["profiles"][phase]["sample_count"] != repeats
            or len(profile_metadata["instrumented_timings"][phase]) != repeats
        ):
            raise ValueError("SQLite profile run count differs from the validated summary")
    for run in runs:
        phase = run["phase"]
        prefix, query = (
            (0, workload["total_tokens"])
            if phase == "full_prefill"
            else (workload["prefix_tokens"], workload["new_tokens"])
        )
        phase_flops = _flops_by_module(config, prefix, query, workload["chunk_size"], cached=cached)
        for call in run["layer_calls"]:
            one_layer = config | {"num_hidden_layers": 1}
            owners = [name for name in QK_MODULES if call["modules"][name]["kernel_count"]]
            scored = (call["query_start"] + call["query_length"] + 63) // 64 > 64
            if len(owners) != int(scored):
                raise ValueError(
                    "Each scored call must own QK in exactly one score or score+selection module"
                )
            owner = owners[0] if owners else None
            if call.get("indexer_qk_module") != owner:
                raise ValueError("Recorded indexer QK owner disagrees with its attributed kernels")
            call_flops = _flops_by_module(
                one_layer,
                call["query_start"],
                call["query_length"],
                workload["chunk_size"],
                cached=cached,
                qk_module=owner,
            )
            for name, module in call["modules"].items():
                _add_mfu(module, call_flops[name], peak)
            length, dim = call["query_length"], config["head_dim"]
            q_width, kv_width = (
                config["num_attention_heads"] * dim,
                config["num_key_value_heads"] * dim,
            )
            call["matrix_shapes_m_n_k"] = {
                "qkv_proj": [length, q_width + 2 * kv_width, config["hidden_size"]],
                "cis_projection": [length, config["num_key_value_heads"], kv_width],
                "o_proj": [length, config["hidden_size"], q_width],
                "gate_up_proj": [length, 2 * config["intermediate_size"], config["hidden_size"]],
                "down_proj": [length, config["hidden_size"], config["intermediate_size"]],
            }
        # Dispatch can mix score-only and fused score+selection within a phase.
        # Sum the actual per-call owners before attaching phase-level MFU.
        for name, module in run["modules"].items():
            flops = (
                sum(call["modules"][name]["matrix_flops"] for call in run["layer_calls"])
                if name != "other_non_matrix"
                else 0
            )
            if name not in QK_MODULES and flops != phase_flops[name]:
                raise ValueError("Layer/chunk matrix FLOPs do not conserve the phase work count")
            _add_mfu(module, flops, peak)
        if (
            sum(run["modules"][name]["matrix_flops"] for name in QK_MODULES)
            != phase_flops["indexer_total"]
        ):
            raise ValueError("Score modules must count each useful indexer QK exactly once")
    phases = {}
    for phase in PHASES:
        selected = [run for run in runs if run["phase"] == phase]
        modules = {}
        for name in MODULES:
            values = [run["modules"][name] for run in selected]
            if len({value["matrix_flops"] for value in values}) != 1:
                raise ValueError("Repeated profile runs disagree on module matrix FLOP ownership")
            kernel_ms_median = statistics.median(value["kernel_ms"] for value in values)
            modules[name] = {
                "sample_count": len(values),
                "parent_module": PARENTS.get(name),
                "inclusive": name in INCLUSIVE_MODULES,
                "matrix_flops_per_run": values[0]["matrix_flops"],
                "kernel_ms_median": kernel_ms_median,
                "kernel_count_per_run": [value["kernel_count"] for value in values],
                "mfu_pct_median": values[0]["matrix_flops"] / kernel_ms_median / 1e9 / peak * 100
                if values[0]["mfu_pct"] is not None
                else None,
            }
            has_unscoped_joint_kernels = name in (
                "pooled_scores",
                "native_selection",
                "score_selection",
            ) and any(
                call.get("native_joint_submission") and call["modules"][name]["kernel_count"]
                for run in selected
                for call in run["layer_calls"]
            )
            if (
                name in summary["profiles"][phase]["stage_totals"]
                and not has_unscoped_joint_kernels
            ):
                interval = summary["profiles"][phase]["stage_totals"][name]["cuda_elapsed_ms"][
                    "median"
                ]
                modules[name]["event_interval_ms_median"] = interval
                modules[name]["event_interval_mfu_pct"] = (
                    values[0]["matrix_flops"] / interval / 1e9 / peak * 100
                    if values[0]["matrix_flops"] and interval > 0
                    else None
                )
        phases[phase] = {"modules": modules, "end_to_end": baseline["phases"][phase]}
    return {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "gpu": baseline["gpu"],
        "peak_tflops": peak,
        "peak_source": baseline["peak_source"],
        "peak_kind": baseline["peak_kind"],
        "workload": workload,
        "implementation": baseline["implementation"],
        "dimensions": baseline["dimensions"],
        "runs": runs,
        "phases": phases,
        "validation": attribution["validation"],
        "definitions": baseline["definitions"]
        | {
            "mfu_pct": "100 * useful matrix FLOPs / (sum of attributed GPU kernel durations in seconds * peak FLOP/s)",
            "kernel_attribution": "Host CUDA launch correlation within NVTX; GPU timestamps supply durations only",
            "native_score_dispatch": "Native QK requires >=2047 compressed keys for >=1024 query rows, otherwise >=511 keys; other scored chunks use one Triton _scores kernel. Reviewed native geometry selects normalizer+scores, fused score-only, or fused score+selection; exact kernel sequence and the Selection template flag are validated per call",
            "score_selection": "The fused score+selection kernel owns one useful indexer QK and its entire combined duration. It is never reported as score-only. FLOPs are assigned per layer/query call before phase aggregation, including mixed full_prefill dispatch",
            "pruned_score_selection": "Revision 5 permits the bounded second-pass pruning kernel only for 1024 queries, 66560 total tokens and two KV heads. Its exact normalizer, bound construction, partial cutoff, surviving QK and final selection all remain in score_selection. The original fused kernel is accepted as the resource-guard fallback. Pruning and recomputation do not change useful QK FLOPs",
            "native_preparation": "Basic and ranked preparation, finite checks and any separate ranking kernels remain within indexer_total. Preparation/ranking has no useful matrix FLOPs; fusion or QK recomputation does not increase useful QK",
            "checked_submission": "The checked native scope is a host container: finite_partials belongs to native_finite_check, guarded_compression_and_ranking to native_ranked_compression, and the remaining exact score/selection sequence to its QK owner. Their kernel durations remain inside indexer_total; pinned flag copies and host synchronization do not add matrix FLOPs or kernel time",
            "native_attention_dispatch": "The reviewed source revision and attention_execution determine the complete launch sequence. native_fa3_v2 uses prepare, FA3 main with direct output and nonfinite detection, and native repair for eligible queries >=1. native_fa3_v3 inserts sort_work_by_union_size between prepare and FA3 exactly when ceil(query_length/8)*num_key_value_heads==256; other positive eligible shapes keep the three-kernel sequence, and empty queries launch nothing. The sorter is included in attention time and adds no useful matrix FLOPs. FP16 or incompatible strides retain the previously reviewed native-sequence alternatives; their stride eligibility is not independently established by this attribution. Grouped native paths include their scan and fallback kernels. Every kernel belongs to one attention module; useful QK/PV FLOPs are counted once per layer/query, independently of launch count or fallback activity",
            "unscoped_gemms": "Reviewed runtime source order [QKV,CIS,O,gate_up,down] plus CIS/attention/RoPE/norm/activation anchors",
            "kernel_shape_proof": "Captured config and source graph, audited resident Q/K/V shapes and complete layer/chunk counts; kernel names/grids are not shape evidence",
            "no_matrix_modules": "Compression, standalone selection and other non-matrix modules have MFU null, not zero",
            "nested_modules": "Indexer children and cis_projection_gemm are inclusive subsets; never add them to their parents",
            "scope_denominator": "CIS/indexer/attention scopes include all correlated kernels, including non-matrix work; cis_projection_gemm is an additional delta-only view",
            "event_interval_metric": "Secondary recorded CUDA-event interval MFU includes submission/launch gaps and is distinct from kernel-duration MFU. No event interval is fabricated for fused score+selection or partial views split from a joint submission",
            "profile_limit": "Kernel metrics come from instrumented nsys runs; end-to-end MFU remains the independent uninstrumented wall-time result",
        },
    }


def _markdown(report):
    lines = [
        "# NOSA module MFU from nsys kernel durations",
        "",
        f"Run: `{report['run_id']}`. Dense BF16 peak: {report['peak_tflops']:g} TFLOP/s.",
        "",
        (
            "Times sum actual correlated GPU kernel durations, not NVTX host ranges or CUDA-event intervals. "
            "Nested modules overlap their parents and must not be added. Compression/selection have no matrix MFU."
        ),
        "",
        "| Phase | Module | Parent | Kernel ms (median) | Kernels/run | Useful matrix FLOPs/run | MFU % |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for phase in PHASES:
        for name, row in report["phases"][phase]["modules"].items():
            mfu = "—" if row["mfu_pct_median"] is None else f"{row['mfu_pct_median']:.4f}"
            lines.append(
                f"| {phase} | {name} | {row['parent_module'] or '—'} | {row['kernel_ms_median']:.6f} | {','.join(map(str, row['kernel_count_per_run']))} | {row['matrix_flops_per_run']} | {mfu} |"
            )
    lines.extend(
        [
            "",
            "Kernel duration sum is not union active time. Timestamp-boundary overlap is audited in JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(data_dir, *, sqlite_path=None, peak_tflops=None, write=True):
    """Derive new output files; leave all captured measurements/source snapshots intact."""
    data_dir = Path(data_dir).resolve()
    sqlite_path = (
        Path(sqlite_path).resolve() if sqlite_path is not None else data_dir / "nsys.sqlite"
    )
    names = ("metadata.json", "summary.json", "profile_metadata.json", "attention_audit.json")
    inputs = {name: (data_dir / name).read_bytes() for name in names}
    metadata, summary, profile_metadata = [json.loads(inputs[name]) for name in names[:3]]
    sources = _validate_inputs(data_dir, metadata, summary, profile_metadata)
    scopes, kernels = read_trace(sqlite_path)
    attribution = attribute_kernels(
        scopes,
        kernels,
        metadata["model_config"]["num_hidden_layers"],
        summary["workload"],
        native_attention_group_size=_validate_source_graph(sources),
        num_kv_heads=metadata["model_config"]["num_key_value_heads"],
    )
    attribution["validation"]["trace_gpu"] = validate_trace_gpu(
        sqlite_path,
        attribution["validation"]["device_id"],
        metadata["gpu"],
        global_pid=attribution["validation"]["global_pid"],
    )
    report = build_module_report(
        metadata, summary, profile_metadata, attribution, peak_tflops=peak_tflops
    )
    root = Path(__file__).resolve().parents[4]
    report["provenance"] = {
        "derived_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in inputs.items()},
        "sqlite": {
            "path": str(sqlite_path),
            "bytes": sqlite_path.stat().st_size,
            "sha256": _sha256(sqlite_path),
        },
        "captured_source_sha256": sources,
        "analysis_source_sha256": {
            name: _sha256(root / name)
            for name in (
                "experiments/nosa_baseline_performance/src/sparse/module_mfu.py",
                "experiments/nosa_baseline_performance/src/sparse/mfu.py",
                "experiments/nosa_baseline_performance/src/sparse/analyze.py",
                "experiments/nosa_baseline_performance/src/dense/mfu.py",
            )
        },
    }
    if write:
        (data_dir / "module_mfu.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        fields = (
            "phase",
            "iteration",
            "module",
            "parent_module",
            "inclusive",
            "kernel_ms",
            "kernel_count",
            "matrix_flops",
            "effective_tflops",
            "mfu_pct",
        )
        with (data_dir / "module_mfu.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for run in report["runs"]:
                for name, row in run["modules"].items():
                    writer.writerow(
                        {
                            field: (
                                {
                                    "phase": run["phase"],
                                    "iteration": run["iteration"],
                                    "module": name,
                                }
                                | row
                            ).get(field)
                            for field in fields
                        }
                    )
        (data_dir / "module_mfu.md").write_text(_markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Completed benchmark/profile run directory")
    parser.add_argument(
        "--sqlite", type=Path, help="nsys SQLite export; default data_dir/nsys.sqlite"
    )
    parser.add_argument(
        "--peak-tflops", type=float, help="Dense BF16 peak; defaults to 989 for H200"
    )
    args = parser.parse_args(argv)
    report = analyze(args.data_dir, sqlite_path=args.sqlite, peak_tflops=args.peak_tflops)
    print(json.dumps({phase: report["phases"][phase]["modules"] for phase in PHASES}, indent=2))


if __name__ == "__main__":
    main()
