#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  echo "Usage: bash experiments/gr_cache_serving/scripts/prepare_echo.sh"
  echo "Prepare the pinned ECHO source only; does not install or run benchmarks."
  exit 0
fi
if [[ $# != 0 ]]; then
  echo "Unexpected arguments; use --help" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
target="$repo_dir/3rdparty/ECHO"
revision=bc1b75c1000010d0ac6f032ebaac283255c050b1
url=https://github.com/sjtu-zhao-lab/ECHO.git

if [[ ! -e "$target" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "$url" "$target"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$target" checkout --detach "$revision"
fi
if [[ ! -d "$target/.git" ]]; then
  echo "Existing ECHO directory is not the expected standalone checkout: $target" >&2
  exit 1
fi
if [[ "$(git -C "$target" rev-parse HEAD)" != "$revision" ]]; then
  echo "Existing ECHO revision differs; preserve it and prepare the pinned revision separately." >&2
  exit 1
fi
if [[ -n "$(git -C "$target" status --porcelain --untracked-files=no)" ]]; then
  echo "ECHO has source changes; record and review the patch before treating it as upstream." >&2
  exit 1
fi
echo "ECHO source verified: $revision"
echo "Path: $target"
echo "This helper does not initialize LFS data or nested dependencies, or certify GPU readiness."
