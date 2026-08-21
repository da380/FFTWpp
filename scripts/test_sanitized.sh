#!/usr/bin/env bash
#
# Build and run the test suite under a sanitizer.
#
#   scripts/test_sanitized.sh [address|thread] [extra cmake args...]
#
# "address" runs AddressSanitizer plus UndefinedBehaviorSanitizer, which
# catches memory errors and the plan-ownership regressions the ownership
# tests are written for. "thread" runs ThreadSanitizer, which is what
# actually demonstrates that the planner mutex serialises FFTW's planner:
# without it, PlannerSerialisation.ConcurrentPlanCreationCompletes reports
# races inside FFTW.
#
# The two cannot be combined, so they are separate runs.

set -euo pipefail

sanitizer="${1:-address}"
if [[ $# -gt 0 ]]; then shift; fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${sanitizer}" in
  address)
    flags="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
    build="${root}/build-asan"
    ;;
  thread)
    flags="-fsanitize=thread -fno-omit-frame-pointer -g"
    build="${root}/build-tsan"
    ;;
  *)
    echo "usage: $0 [address|thread] [extra cmake args...]" >&2
    exit 2
    ;;
esac

cmake -S "${root}" -B "${build}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="${flags}" \
  -DCMAKE_EXE_LINKER_FLAGS="${flags}" \
  "$@"
cmake --build "${build}" --parallel

# FFTW itself is linked uninstrumented, so ThreadSanitizer cannot see the
# happens-before edges inside it. In practice it stays quiet, so no
# suppression file is shipped; set TSAN_SUPPRESSIONS to one if a particular
# FFTW build turns out to need it. Note that a called_from_lib entry must
# match exactly one loaded library or ThreadSanitizer refuses to start, so
# name the full soname rather than "libfftw3".
export TSAN_OPTIONS="halt_on_error=1:second_deadlock_stack=1${TSAN_SUPPRESSIONS:+:suppressions=${TSAN_SUPPRESSIONS}}"
export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
export UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1"

# GCC's ThreadSanitizer refuses to start when the kernel's mmap randomisation
# is wider than it expects, which is the default on recent Linux. Running with
# ASLR disabled avoids it; the personality is inherited by the test binaries.
launcher=()
if [[ "${sanitizer}" == "thread" ]] && command -v setarch >/dev/null; then
  launcher=(setarch "$(uname -m)" -R)
fi

"${launcher[@]}" ctest --test-dir "${build}" --output-on-failure
