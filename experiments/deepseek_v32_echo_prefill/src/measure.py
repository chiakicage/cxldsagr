"""Three-layer checkpoint benchmark entry and retained measurement utilities."""

import hashlib
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from tokenizers import Tokenizer

from experiments.deepseek_v32_echo_prefill.src.backend_provenance import source_files
from GR.input_generator import INSTRUCTION, TextConfig, create_input_generator
from GR.scheduling import ScheduleConfig
from models.deepseek_v32 import request_format


class Scopes:
    def __init__(self, phase):
        self.phase = phase
        self.events = []
        self.layer = "shared"
        self.stack = []

    @contextmanager
    def __call__(self, name):
        if name.startswith("layer_"):
            previous, self.layer = self.layer, name
            try:
                with torch.cuda.nvtx.range(f"echo/{self.phase}/{name}"):
                    yield
            finally:
                self.layer = previous
            return
        label = f"echo/{self.phase}/{self.layer}/{name}"
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        node = {
            "stage": name,
            "layer": self.layer,
            "device": torch.cuda.current_device(),
            "start": start,
            "stop": stop,
            "children": [],
        }
        if self.stack:
            self.stack[-1]["children"].append(node)
        self.stack.append(node)
        with torch.cuda.nvtx.range(label), torch.profiler.record_function(label):
            start.record()
            begin = time.perf_counter()
            try:
                yield
            finally:
                node["host_ms"] = (time.perf_counter() - begin) * 1000
                stop.record()
                self.stack.pop()
                self.events.append(node)

    def summary(self):
        totals = defaultdict(float)
        rows = []
        for node in self.events:
            elapsed = node["start"].elapsed_time(node["stop"])
            child_ms = sum(x["start"].elapsed_time(x["stop"]) for x in node["children"])
            exclusive = max(0, elapsed - child_ms)
            totals[node["stage"]] += exclusive
            rows.append(
                {k: node[k] for k in ("stage", "layer", "device", "host_ms")}
                | {
                    "cuda_inclusive_ms": elapsed,
                    "cuda_exclusive_ms": exclusive,
                }
            )
        return {"stages_cuda_exclusive_ms": dict(totals), "stage_calls": rows}


def make_request(model, prefix, extend, seed):
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    overhead = len(
        tokenizer.encode(request_format.prefix(INSTRUCTION), add_special_tokens=False).ids
    )
    if prefix <= overhead:
        raise ValueError("prefix must accommodate the GR instruction")
    generator = create_input_generator(
        model="deepseek_v32",
        tokenizer=tokenizer,
        num_users=1000,
        text_config=TextConfig(
            user_lengths=(prefix - overhead,),
            user_probabilities=(1.0,),
            item_lengths=(extend + overhead,),
            item_probabilities=(1.0,),
            max_input_tokens=prefix + extend,
        ),
        schedule_config=ScheduleConfig(seed=seed, sampling="weighted"),
    )
    request = next(generator.iter_generate(1))
    if request["stable_prefix_tokens"] != prefix or request["candidate_suffix_tokens"] != extend:
        raise ValueError("GR semantic boundary differs from the measured execution boundary")
    return request


def source_manifest():
    root = Path(__file__).resolve().parents[3]
    paths = [
        *source_files(),
        *root.glob("models/deepseek_v32/echo_*.py"),
        root / "models/deepseek_v32/nonmatrix.py",
        root / "operators/flashinfer.py",
        root / "models/deepseek_v32/request_format.py",
        *(
            path
            for directory in ("operators/deepseek_v32", "operators/common")
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "tests" not in path.relative_to(root / directory).parts
            and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
        ),
        *root.glob("cache/*.py"),
        root / "models/deepseek_v32/cache_resources.py",
        Path(__file__).resolve(),
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def timed(model, ids, label, *, annotate=False):
    model.synchronize()
    scopes = Scopes(label) if annotate else None
    begin = time.perf_counter()
    output = model.forward(ids, scope=scopes)
    elapsed = (time.perf_counter() - begin) * 1000
    if not torch.isfinite(output).all():
        raise RuntimeError(f"nonfinite full-model output for {label}")
    return output, {"wall_ms": elapsed, **(scopes.summary() if scopes else {})}


def main():
    """Use the same real three-layer workload and controls as the profile entry."""
    from experiments.deepseek_v32_echo_prefill.src.profile_layers import main as profile_main

    profile_main()


if __name__ == "__main__":
    main()
