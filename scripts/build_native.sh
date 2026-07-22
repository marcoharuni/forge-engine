#!/usr/bin/env bash
set -euo pipefail

build_dir="${FORGE_NATIVE_BUILD_DIR:-.tmp/native-build}"
cmake -S native -B "${build_dir}" -DFORGE_ENABLE_CUDA="${FORGE_ENABLE_CUDA:-OFF}"
cmake --build "${build_dir}"

