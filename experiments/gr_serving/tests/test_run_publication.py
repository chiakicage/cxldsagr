"""A failed independent audit must never publish a controlled GR run."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("audit_exit", [0, 17])
@pytest.mark.parametrize("output_root", [None, "experiments/deepseek_v32_echo_cache/output"])
def test_controlled_run_audits_staging_before_publication(tmp_path, audit_exit, output_root):
    root = tmp_path / "repo"
    scripts = root / "experiments/gr_serving/scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "scripts/run.sh", scripts / "run.sh")
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[1].endswith('.measure'):\n"
        "    target = pathlib.Path(args[args.index('--output-dir') + 1])\n"
        "    target.mkdir()\n"
        "    (target / 'metadata.json').write_text('{}')\n"
        "elif args[1].endswith('.audit'):\n"
        "    target = pathlib.Path(args[args.index('--json') + 1])\n"
        "    assert 'experiments' not in target.parts\n"
        "    code = int(os.environ['FAKE_AUDIT_EXIT'])\n"
        "    if code: sys.exit(code)\n"
        "    target.write_text('{}')\n"
        "else: raise AssertionError(args)\n"
    )
    python.chmod(0o755)
    temporary = tmp_path / "staging"
    temporary.mkdir()
    result = subprocess.run(
        ["bash", str(scripts / "run.sh"), "test_run"],
        env={
            **os.environ,
            "TMPDIR": str(temporary),
            "GR_AUDIT_BEFORE_PUBLISH": "1",
            "FAKE_AUDIT_EXIT": str(audit_exit),
            "GR_OUTPUT_ROOT": output_root or "",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == audit_exit, result.stderr
    output = root / (output_root or "experiments/gr_serving/output")
    if audit_exit:
        assert not output.exists()
        assert len(list(temporary.glob("*/data/metadata.json"))) == 1
        assert "diagnostics remain outside experiments" in result.stderr
    else:
        assert (output / "data/test_run/independent_audit.json").is_file()
        assert not list(temporary.iterdir())
