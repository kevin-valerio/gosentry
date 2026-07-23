#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/misc/gosentry/tests/smoke_use_libafl_common.sh"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT
export GOCACHE="${tmp_dir}/gocache"

cfg_path="${tmp_dir}/libafl-config.jsonc"
cat >"${cfg_path}" <<'EOF'
{
  "cores": "0",
  "stop_all_fuzzers_on_panic": false,
  "tui_monitor": false,
  "debug_output": true
}
EOF

cd "${ROOT_DIR}/test/gosentry/examples/sigsegv_restart"
output_file="${tmp_dir}/output.txt"
set +e
GOSENTRY_VERBOSE_AFL=1 CGO_ENABLED=1 timeout 10m "${ROOT_DIR}/bin/go" test -fuzz=FuzzSIGSEGV --use-libafl --focus-on-new-code=false --catch-races=false --catch-leaks=false -parallel=1 -fuzztime=2s --libafl-config="${cfg_path}" . 2>&1 | tee "${output_file}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" -ne 0 ]]; then
  echo "expected fuzz run to continue after SIGSEGV, got exit ${status}"
  exit 1
fi

if ! grep -Eq "${GOSENTRY_LIBAFL_CRASH_RE}" "${output_file}"; then
  echo "expected LibAFL to save the SIGSEGV input as a crash"
  exit 1
fi

if grep -Fq "The fuzzer crashed inside a crash handler" "${output_file}"; then
  echo "LibAFL mistook the target SIGSEGV for a crash-handler failure"
  exit 1
fi

restart_count="$(grep -Fc "golibafl: client start id=1" "${output_file}" || true)"
if [[ "${restart_count}" -lt 2 ]]; then
  echo "expected LibAFL to restart its client after SIGSEGV"
  exit 1
fi
