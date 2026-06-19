#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

cd "${ROOT_DIR}/test/gosentry/examples/fuzztime"

set +e
CGO_ENABLED=1 timeout 2m "${ROOT_DIR}/bin/go" test -fuzz=FuzzSomeFunc -fuzztime=2s -focus-on-new-code=false -catch-races=false -catch-leaks=false 2>&1 | tee "${tmp_dir}/output.txt"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "expected -fuzztime LibAFL run to exit 0, got ${status}"
  exit 1
fi

if ! grep -Eq '^ok[[:space:]]+fuzztime[[:space:]]+' "${tmp_dir}/output.txt"; then
  echo "expected go test ok line for fuzztime package"
  exit 1
fi

if ! grep -Fq "Fuzzing stopped by user" "${tmp_dir}/output.txt"; then
  echo "expected LibAFL runner to start and stop at -fuzztime"
  exit 1
fi
