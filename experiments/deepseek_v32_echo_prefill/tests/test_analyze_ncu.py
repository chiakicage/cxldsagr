"""CPU fakes exercise report extraction without CUDA or native NCU imports."""

import json
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_prefill.src import analyze_ncu


class Metric:
    ValueKind_UNKNOWN = 0
    ValueKind_STRING = 2
    ValueKind_DOUBLE = 4
    ValueKind_UINT64 = 6
    RollupOperation_SUM = 4

    def __init__(
        self, value=None, *, values=(), ids=None, unit="sample", name="fixture", failures=()
    ):
        self.aggregate = value
        self.values = list(values)
        self.ids = ids
        self.units = unit
        self.metric_name = name
        self.failures = set(failures)

    def name(self):
        return self.metric_name

    def unit(self):
        return self.units

    def description(self):
        return "CPU fake metric"

    def kind(self):
        if self.aggregate is None:
            return self.ValueKind_UNKNOWN
        if isinstance(self.aggregate, int):
            return self.ValueKind_UINT64
        if isinstance(self.aggregate, str):
            return self.ValueKind_STRING
        return self.ValueKind_DOUBLE

    def rollup_operation(self):
        return self.RollupOperation_SUM

    def num_instances(self):
        return len(self.values)

    def has_value(self, kind=None):
        # The native optional argument is ValueKind, not an instance index.
        return self.aggregate is not None and kind in (None, 1, self.kind())

    def value(self, index=None):
        if index in self.failures:
            raise RuntimeError("fixture read failure")
        return self.aggregate if index is None else self.values[index]

    def has_correlation_ids(self):
        return self.ids is not None

    def correlation_ids(self):
        return Metric(values=self.ids, unit="opaque-id", name="correlation_ids")


class Action:
    NameBase_DEMANGLED = 1
    NameBase_MANGLED = 2
    WorkloadType_KERNEL = 0

    def __init__(self, name="kernel", metrics=None, source=None):
        self.kernel_name = name
        self.metrics = metrics or {}
        self.source = source or {}
        self.pcs_requested = []

    def name(self, base=None):
        return self.kernel_name + {None: "", 1: "(float*)", 2: "_mangled"}[base]

    def workload_type(self):
        return self.WorkloadType_KERNEL

    def metric_names(self):
        return tuple(self.metrics)

    def metric_by_name(self, name):
        value = self.metrics[name]
        if isinstance(value, Exception):
            raise value
        return value

    def source_info(self, pc):
        self.pcs_requested.append(pc)
        location = self.source.get(pc)
        if isinstance(location, Exception):
            raise location
        if location is None:
            return None
        file_name, line = location
        return SimpleNamespace(file_name=lambda: file_name, line=lambda: line)

    def source_files(self):
        return {"kernel.cu": "line one\nline two\nline three", "unavailable.cu": ""}

    def sass_by_pc(self, pc):
        return f"LDG.E {pc}"


class ReportRange:
    def __init__(self, *actions):
        self.actions = actions

    def num_actions(self):
        return len(self.actions)

    def action_by_idx(self, index):
        return self.actions[index]


class Report:
    def __init__(self, *ranges):
        self.ranges = ranges

    def num_ranges(self):
        return len(self.ranges)

    def range_by_idx(self, index):
        return self.ranges[index]


def test_all_ranges_actions_names_and_metric_types_are_retained(tmp_path):
    action = Action(
        "first_kernel<128>",
        {
            "gpu__time_duration.sum": Metric(12.75, unit="nsecond"),
            "launch__block_size": Metric(256, unit="thread"),
            "custom__uint64": Metric(2**63 + 91, unit="byte"),
            "custom__label": Metric("device label", unit=""),
        },
    )
    report = Report(
        ReportRange(action, Action("second_kernel")), ReportRange(Action("third_kernel"))
    )
    result = analyze_ncu.analyze_report(report, tmp_path / "input.ncu-rep")
    assert result["range_count"] == 2
    assert result["action_count"] == 3
    first, second = result["ranges"][0]["actions"]
    assert first["name"] == "first_kernel<128>(float*)"
    assert second["action_index"] == 1
    assert result["ranges"][1]["actions"][0]["names"]["function"] == "third_kernel"
    metrics = first["metrics"]
    assert len(metrics) == 4
    assert metrics["gpu__time_duration.sum"]["value"] == 12.75
    assert metrics["gpu__time_duration.sum"]["unit"] == "nsecond"
    assert metrics["custom__uint64"]["value"] == 2**63 + 91
    assert metrics["custom__uint64"]["kind"]["name"] == "UINT64"
    assert metrics["custom__label"]["value"] == "device label"
    assert "launch__block_size" in first["key_metrics"]
    assert "dram__bytes_read.sum" in first["absent_key_metric_candidates"]
    assert not first["errors"]


def test_pm_preserves_order_uint64_zero_missing_nonfinite_and_statistics():
    ids = [2**63 + 7, 2**63 + 1, 2**63 + 1, None, 0, 3, 4]
    values = [0, 6.5, None, float("nan"), float("inf"), float("-inf"), 1.5]
    name = "pmsampling:sm__throughput.avg.pct_of_peak_sustained_elapsed"
    result = analyze_ncu.analyze_action(
        Action(metrics={name: Metric(values=values, ids=ids, unit="%")})
    )
    series = result["pm_sampling"][name]
    # Strict JSON roundtrip preserves every integer bit and distinguishes missing/nonfinite.
    series = json.loads(json.dumps(series, allow_nan=False))
    assert series["correlation_ids"] == ids
    assert series["correlation_metadata"]["unit"] == "opaque-id"
    assert series["values"] == [
        0,
        6.5,
        None,
        {"nonfinite": "NaN"},
        {"nonfinite": "Infinity"},
        {"nonfinite": "-Infinity"},
        1.5,
    ]
    assert series["value_status"] == [
        "present",
        "present",
        "missing",
        "nonfinite",
        "nonfinite",
        "nonfinite",
        "present",
    ]
    stats = series["statistics"]
    assert stats["count"] == 7
    assert stats["finite_numeric_count"] == 3
    assert stats["zero_count"] == 1
    assert stats["missing_count"] == 1
    assert stats["nonfinite_count"] == 3
    assert stats["mean"] == pytest.approx(8 / 3)
    assert stats["population_stddev"] == pytest.approx(math.sqrt(139 / 18))


def test_native_valuekind_presence_does_not_filter_instance_indices():
    name = "TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg"
    values = [0, 11, 22, 33, 44, 55, 66, 77]
    metric = Metric(sum(values), values=values, ids=list(range(100, 108)))
    result = analyze_ncu.analyze_action(Action(metrics={name: metric}))
    series = result["pm_sampling"][name]
    assert series["values"] == values
    assert series["correlation_ids"] == list(range(100, 108))
    assert series["statistics"]["finite_numeric_count"] == 8
    assert not series["errors"]


def test_pc_source_join_and_per_metric_sums_do_not_mix_samples_and_stalls():
    count = "smsp__pcsamp_sample_count"
    stall = "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
    ratio = stall + ".pct"
    ids = [16, 32, 48, 64]
    action = Action(
        metrics={
            count: Metric(103, values=[10, 20, 30, 43], ids=ids),
            stall: Metric(18, values=[4, 6, 0, 8], ids=ids),
            ratio: Metric(20.0, values=[50.0, 25.0], ids=ids[:2], unit="%"),
        },
        source={16: ("kernel.cu", 2), 32: ("kernel.cu", 2), 48: ("kernel.cu", 3)},
    )
    pc = analyze_ncu.analyze_action(action)["pc_sampling"]
    assert sorted(action.pcs_requested) == ids  # Each PC source resolved once across metrics.
    assert pc["source_by_pc"]["16"]["source_text"] == "line two"
    assert pc["source_by_pc"]["16"]["sass"] == "LDG.E 16"
    assert pc["source_by_pc"]["64"]["source_known"] is False
    assert pc["source_by_pc"]["64"]["line"] is None
    assert pc["metrics"][count]["source_lines"][0]["sample_sum"] == 30
    stalls = pc["metrics"][stall]
    assert stalls["source_lines"][0]["sample_sum"] == 10
    assert stalls["source_lines"][1]["sample_sum"] == 0
    assert stalls["source_lines"][0]["pcs"] == [16, 32]
    assert stalls["unmapped_statistics"]["maximum"] == 8
    assert pc["metrics"][ratio]["source_lines"][0]["sample_sum"] is None


def test_missing_metrics_getter_failure_and_unknown_source_are_explicit():
    pm = "pmsampling:broken"
    stall = "smsp__pcsamp_warps_issue_stalled_wait"
    result = analyze_ncu.analyze_action(
        Action(
            metrics={
                "gone": None,
                "getter_error": RuntimeError("metric unavailable"),
                pm: Metric(7, values=[5, 2], ids=[33], failures=(0,)),
                stall: Metric(values=[9, None], ids=[99, 100]),
            },
            source={99: RuntimeError("no source table"), 100: ("unavailable.cu", 4)},
        )
    )
    assert result["unavailable_metric_names"] == ["gone", "getter_error"]
    assert result["metrics"]["getter_error"]["errors"][0]["method"] == "metric_by_name"
    series = result["pm_sampling"][pm]
    assert series["values"] == [None, 2]
    assert series["value_status"] == ["error", "present"]
    assert series["correlation_ids"] == [33]  # Length mismatch never truncates either array.
    assert series["correlation_count_matches_values"] is False
    assert series["errors"][0]["args"] == [0]
    source = result["pc_sampling"]["source_by_pc"]
    assert source["99"]["source_known"] is False
    assert source["99"]["errors"][0]["method"] == "source_info"
    assert source["100"]["source_known"] is True
    assert source["100"]["source_text"] is None
    line = result["pc_sampling"]["metrics"][stall]["source_lines"][0]
    assert line["sample_sum"] is None
    assert line["sample_sum_complete"] is False


def test_absent_correlation_ids_do_not_fabricate_pc_or_pm_times():
    stall = "smsp__pcsamp_warps_issue_stalled_wait"
    result = analyze_ncu.analyze_action(
        Action(metrics={stall: Metric(values=[0, 2]), "pmsampling:other": Metric(values=[1, 0])})
    )
    pc = result["pc_sampling"]
    assert pc["source_by_pc"] == {}
    assert pc["metrics"][stall]["correlation_ids"] is None
    assert pc["metrics"][stall]["unmapped_statistics"]["count"] == 2
    assert result["pm_sampling"]["pmsampling:other"]["correlation_metadata"] is None


def test_source_files_accepts_native_swig_mapping_without_dict_get():
    class SourceMap:
        def keys(self):
            return ["kernel.cu"]

        def __getitem__(self, key):
            assert key == "kernel.cu"
            return "report's embedded source"

    class NativeMappingAction(Action):
        def source_files(self):
            return SourceMap()

    action = NativeMappingAction(
        metrics={"smsp__pcsamp_sample_count": Metric(values=[1], ids=[16])},
        source={16: ("kernel.cu", 1)},
    )
    result = analyze_ncu.analyze_action(action)
    assert result["pc_sampling"]["source_by_pc"]["16"]["source_text"] == "report's embedded source"


def test_loading_multiple_reports_and_cli_output_never_overwrites(tmp_path, monkeypatch):
    inputs = [tmp_path / "full.ncu-rep", tmp_path / "source.ncu-rep"]
    for path in inputs:
        path.write_bytes(b"fake report")
    loaded = []

    def load_report(path):
        loaded.append(path)
        return Report(ReportRange(Action(path)))

    monkeypatch.setattr(
        analyze_ncu,
        "_load_api",
        lambda _: SimpleNamespace(load_report=load_report, __file__="fake API"),
    )
    output = tmp_path / "data/run_id/ncu_analysis.json"
    arguments = ["--report", str(inputs[0]), "--report", str(inputs[1]), "--output", str(output)]
    analyze_ncu.main(arguments)
    result = json.loads(output.read_text())
    assert len(loaded) == 2
    assert len(result["reports"]) == 2
    assert result["ncu_report_module"] == "fake API"
    assert len(result["reports"][0]["sha256"]) == 64
    assert [path.read_bytes() for path in inputs] == [b"fake report", b"fake report"]
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        analyze_ncu.main(arguments)
    assert output.read_bytes() == original
    assert len(loaded) == 2
    with pytest.raises(SystemExit, match="2"):
        analyze_ncu.main(["--report", str(inputs[0]), "--output", str(inputs[0])])


def test_missing_range_or_action_fails_instead_of_silently_dropping_it(tmp_path):
    for report in (Report(None), Report(ReportRange(None))):
        with pytest.raises(ValueError, match="Missing"):
            analyze_ncu.analyze_report(report, tmp_path / "fixture.ncu-rep")


def test_cli_help_requires_only_python_standard_library():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "experiments.deepseek_v32_echo_prefill.src.analyze_ncu",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--report" in result.stdout
    assert "--run-id" in result.stdout
    assert "--ncu-python-path" in result.stdout
