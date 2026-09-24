"""Initialize flat submodules and connect the upstream include paths."""

import argparse
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "3rdparty"
DEPENDENCIES = {"cutlass": "cutlass", "deep_jit": "DeepJIT"}


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def prepare():
    consumer = THIRD_PARTY / "DeepGEMM"
    if not (consumer / ".git").exists():
        raise RuntimeError("DeepGEMM is not initialized; run this script with --init")
    links = []
    for upstream_name, shared_name in DEPENDENCIES.items():
        shared = THIRD_PARTY / shared_name
        nested = consumer / "third-party" / upstream_name
        if not (shared / ".git").exists():
            raise RuntimeError(f"{shared_name} is not initialized; run with --init")
        expected = git(consumer, "ls-tree", "HEAD", f"third-party/{upstream_name}").split()
        if len(expected) < 3 or expected[0] != "160000":
            raise RuntimeError(f"Upstream dependency layout changed: {upstream_name}")
        if git(shared, "rev-parse", "HEAD") != expected[2]:
            raise RuntimeError(f"{shared_name} must match DeepGEMM's pin {expected[2]}")
        if (nested / ".git").exists():
            raise RuntimeError(f"Nested checkout at {nested}; use only the top-level submodules")
        target = shared / "include"
        link = nested / "include"
        if not target.is_dir():
            raise RuntimeError(f"Missing dependency headers: {target}")
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Existing symlink points elsewhere: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing files: {link}")
        links.append((upstream_name, link, target))

    for upstream_name, link, target in links:
        # Keep upstream gitlinks uninitialized; only their include path is populated.
        git(consumer, "config", f"submodule.third-party/{upstream_name}.active", "false")
        git(consumer, "config", f"submodule.third-party/{upstream_name}.update", "none")
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        print(f"{link.relative_to(ROOT)} -> {os.readlink(link)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="Initialize top-level submodules first")
    args = parser.parse_args()
    if args.init:
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "-c",
                "submodule.recurse=false",
                "submodule",
                "update",
                "--init",
            ],
            check=True,
        )
    try:
        prepare()
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
