"""Collect profiling hardware evidence through NVML's CLI, without importing CUDA.

Device selectors are physical NVIDIA-SMI indices or UUIDs, unaffected by
CUDA_VISIBLE_DEVICES. Runtime CUDA properties belong in the caller's metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.device_id",
    "pci.bus_id",
    "compute_cap",
    "memory.total",
    "clocks.max.sm",
    "clocks.current.sm",
    "power.limit",
    "utilization.gpu",
    "driver_version",
)
_SPEC_URL = "https://www.nvidia.com/en-us/data-center/h200/"


class _TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.text = [], []
        self.table, self.row, self.cell = None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("th", "td") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)


def parse_h200_specification(html):
    """Extract the official SXM column and sparsity footnote, preserving evidence."""
    reader = _TableReader()
    reader.feed(html)
    candidates = [
        table for table in reader.tables if table and any("H200 SXM" in x for x in table[0])
    ]
    if len(candidates) != 1:
        raise ValueError("official page must contain exactly one H200 SXM specification table")
    table = candidates[0]
    sxm_column = next(index for index, cell in enumerate(table[0]) if "H200 SXM" in cell)
    selected = {}
    for label in ("BFLOAT16 Tensor Core", "FP8 Tensor Core", "FP32"):
        matches = [row for row in table if row and row[0].startswith(label)]
        if len(matches) != 1 or len(matches[0]) <= sxm_column:
            raise ValueError(f"official page lacks an unambiguous {label} SXM row")
        row = matches[0]
        number = re.fullmatch(r"([\d,]+(?:\.\d+)?)\s+TFLOPS", row[sxm_column])
        if number is None:
            raise ValueError(f"unexpected TFLOPS format: {row[sxm_column]}")
        selected[label] = {"label": row[0], "published_tflops": float(number[1].replace(",", ""))}
    text = " ".join(" ".join(reader.text).split())
    footnote = re.search(r"2\s+With sparsity\.", text)
    if footnote is None:
        raise ValueError("official page lacks the numeric With sparsity footnote")
    for label in ("BFLOAT16 Tensor Core", "FP8 Tensor Core"):
        if "²" not in selected[label]["label"]:
            raise ValueError(f"{label} is not explicitly linked to sparsity footnote 2")
    if selected["FP32"]["label"] != "FP32":
        raise ValueError("FP32 row unexpectedly carries a footnote")
    return {
        "product_column": table[0][sxm_column],
        "specification_table": table,
        "selected_rows": selected,
        "sparsity_footnote": footnote[0],
        "dense_peaks_tflops": {
            "BF16": selected["BFLOAT16 Tensor Core"]["published_tflops"] / 2,
            "FP8": selected["FP8 Tensor Core"]["published_tflops"] / 2,
            "FP32": selected["FP32"]["published_tflops"],
        },
        "conversion": "BF16 and FP8 published sparsity peaks divided by two; FP32 unchanged",
    }


def _pci_identity(combined_id, candidates=None):
    combined = int(combined_id, 16)
    vendor, device = f"{combined & 0xFFFF:04x}", f"{combined >> 16:04x}"
    candidates = candidates or (Path("/usr/share/misc/pci.ids"), Path("/usr/share/hwdata/pci.ids"))
    path = next((Path(path) for path in candidates if Path(path).is_file()), None)
    evidence = {
        "vendor_id": vendor,
        "device_id": device,
        "database_path": None,
        "matched_line": None,
    }
    if path is None:
        return evidence
    data = path.read_bytes()
    lines = data.decode(errors="replace").splitlines()
    active = False
    for line in lines:
        if line.startswith(vendor + " "):
            active = True
        elif active and line and not line.startswith(("\t", "#")):
            break
        elif active and re.match(rf"\t{device}\s", line):
            evidence["matched_line"] = line
            break
    evidence.update(
        database_path=str(path),
        database_sha256=hashlib.sha256(data).hexdigest(),
        database_version=next((line for line in lines if "Version:" in line), None),
    )
    return evidence


def gather_hardware(device: int | str = 0) -> dict:
    """Return physical GPU identity, raw query evidence and verified reference peaks."""
    command = [
        "nvidia-smi",
        "--id=" + str(device),
        "--query-gpu=" + ",".join(_FIELDS),
        "--format=csv",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    parsed = list(csv.reader(io.StringIO(completed.stdout), skipinitialspace=True))
    if len(parsed) != 2 or len(parsed[1]) != len(_FIELDS):
        raise ValueError("GPU selector must identify exactly one device with all requested fields")
    values = dict(zip(_FIELDS, (value.strip() for value in parsed[1])))
    evidence = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "physical_device_selector": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selector_semantics": "physical nvidia-smi index or UUID; CUDA_VISIBLE_DEVICES does not remap it",
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "query_headers": parsed[0],
        "gpu": values,
        "is_sm90": values["compute_cap"] == "9.0",
        "pci_identity": _pci_identity(values["pci.device_id"]),
        "hardware_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    spec = {"url": _SPEC_URL, "verified": False}
    try:
        request = urllib.request.Request(
            _SPEC_URL, headers={"User-Agent": "ECHO-profile-evidence/1"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            html = response.read()
            spec["http_status"] = response.status
            spec["resolved_url"] = response.url
        spec.update(
            retrieved_at_utc=datetime.now(UTC).isoformat(),
            response_sha256=hashlib.sha256(html).hexdigest(),
            response_bytes=len(html),
        )
        spec.update(parse_h200_specification(html.decode("utf-8")))
        spec["verified"] = True
    except (OSError, UnicodeError, ValueError) as error:
        spec["error"] = f"{type(error).__name__}: {error}"
    evidence["h200_sxm_reference"] = spec
    identity = evidence["pci_identity"]
    evidence["h200_sxm_reference_matches_pci"] = (
        identity["vendor_id"] == "10de"
        and identity["device_id"] == "2335"
        and identity["matched_line"] is not None
        and "H200 SXM" in identity["matched_line"]
    )
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="physical NVIDIA-SMI index or GPU UUID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh evidence path")
    result = gather_hardware(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "gpu": result["gpu"]}))


if __name__ == "__main__":
    main()
