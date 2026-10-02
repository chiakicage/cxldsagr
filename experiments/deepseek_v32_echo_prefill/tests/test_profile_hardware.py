"""Verify SKU identity and sparse-to-dense peak interpretation without a GPU."""

import io
from types import SimpleNamespace

import pytest

from experiments.deepseek_v32_echo_prefill.src import profile_hardware

SPEC = """
<table><tr><th></th><th>H200 SXM¹</th><th>H200 NVL¹</th></tr>
<tr><td>BFLOAT16 Tensor Core²</td><td>1,979 TFLOPS</td><td>1,671 TFLOPS</td></tr>
<tr><td>FP8 Tensor Core²</td><td>3,958 TFLOPS</td><td>3,341 TFLOPS</td></tr>
<tr><td>FP32</td><td>67 TFLOPS</td><td>60 TFLOPS</td></tr></table>
<p><sup>2</sup> With sparsity.</p>
"""


def test_peaks_select_sxm_and_halve_only_explicit_sparsity_rows():
    result = profile_hardware.parse_h200_specification(SPEC)
    assert result["dense_peaks_tflops"] == {"BF16": 989.5, "FP8": 1979.0, "FP32": 67.0}
    assert result["selected_rows"]["FP8 Tensor Core"]["published_tflops"] == 3958
    assert result["sparsity_footnote"] == "2 With sparsity."
    with pytest.raises(ValueError, match="footnote"):
        profile_hardware.parse_h200_specification(
            SPEC.replace("With sparsity.", "Footnote missing")
        )
    with pytest.raises(ValueError, match="explicitly linked"):
        profile_hardware.parse_h200_specification(
            SPEC.replace("FP8 Tensor Core²", "FP8 Tensor Core")
        )


def test_pci_database_matches_vendor_then_device_and_retains_original_evidence(tmp_path):
    path = tmp_path / "pci.ids"
    path.write_text(
        "# Version: fixture\n1000  Other vendor\n\t2335  Other GPU\n"
        "10de  NVIDIA\n\t2335  GH100 [H200 SXM 141GB]\n2000  Last vendor\n"
    )
    result = profile_hardware._pci_identity("0x233510DE", [path])
    assert result["vendor_id"] == "10de" and result["device_id"] == "2335"
    assert result["matched_line"] == "\t2335  GH100 [H200 SXM 141GB]"
    assert len(result["database_sha256"]) == 64
    assert profile_hardware._pci_identity("0x999910DE", [path])["matched_line"] is None


def test_gather_uses_physical_selector_and_readonly_cli_despite_visible_device_mapping(monkeypatch):
    commands = []
    raw = (
        ",".join(profile_hardware._FIELDS) + "\n"
        "1,NVIDIA M403,GPU-physical,0x233510DE,00000000:2A:00.0,9.0,"
        "143771 MiB,1980 MHz,1980 MHz,700 W,0 %,570.124.06\n"
    )

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"check": True, "capture_output": True, "text": True, "timeout": 20}
        return SimpleNamespace(stdout=raw, stderr="")

    class Response(io.BytesIO):
        status = 200
        url = profile_hardware._SPEC_URL

    monkeypatch.setattr(profile_hardware.subprocess, "run", run)
    monkeypatch.setattr(
        profile_hardware.urllib.request, "urlopen", lambda *a, **k: Response(SPEC.encode())
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    result = profile_hardware.gather_hardware(1)
    assert commands[0][1] == "--id=1"
    assert result["cuda_visible_devices"] == "5"
    assert result["gpu"]["uuid"] == "GPU-physical"
    assert result["stdout"] == raw
    assert result["is_sm90"]
    assert result["h200_sxm_reference"]["verified"]


def test_network_failure_keeps_hardware_and_never_claims_verified_peaks(monkeypatch):
    raw = ",".join(profile_hardware._FIELDS) + "\n0,GPU,UUID,0x233510DE,PCI,9.0,1,1,1,1,0,v\n"
    monkeypatch.setattr(
        profile_hardware.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=raw, stderr=""),
    )

    def fail(*args, **kwargs):
        raise TimeoutError("offline fixture")

    monkeypatch.setattr(profile_hardware.urllib.request, "urlopen", fail)
    result = profile_hardware.gather_hardware("GPU-physical")
    assert result["is_sm90"]
    assert not result["h200_sxm_reference"]["verified"]
    assert "offline fixture" in result["h200_sxm_reference"]["error"]
    assert "dense_peaks_tflops" not in result["h200_sxm_reference"]
