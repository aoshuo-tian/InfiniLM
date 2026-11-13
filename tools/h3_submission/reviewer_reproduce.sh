#!/usr/bin/env bash
# Explicit stages only. This script never allocates platform resources.
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CORE="$(dirname -- "$REPO")/InfiniCore"
TOOLS="$REPO/tools/h3_submission"
: "${H3_REPRO_ROOT:?Set H3_REPRO_ROOT to a persistent writable directory outside the code checkouts}"
WORK="$H3_REPRO_ROOT/build"
RESULTS="${H3_OUTPUT_ROOT:-$H3_REPRO_ROOT/results}"
ASSETS="${H3_ASSETS:-$H3_REPRO_ROOT/assets}"
MODEL="${H3_MODEL:-/public/home/aclsylqidf/shared/MiniMax-H3/diffusers}"
IMAGE="$REPO/examples/h3_submission/tp8_20261109/profile-sources/user-puppy-clear-8dacfbb2.jpg"
STAGE="${1:-help}"

load_build() {
    source "$WORK/setup.sh"
}

flash_arguments() {
    FLASH_SO="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["flash_so"])' "$WORK/vendor-flash/prepare.json")"
    FLASH_SHA256="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["flash_so_sha256"])' "$WORK/vendor-flash/prepare.json")"
}

worker_arguments() {
    python "$TOOLS/install_dependencies.py" --work "$WORK" --check-only
    flash_arguments
    WORKER_ARGS=(--work "$WORK" --model "$MODEL" --first-frame "$IMAGE"
                 --flash-so "$FLASH_SO" --flash-sha256 "$FLASH_SHA256")
    mkdir -p "$RESULTS"
}

case "$STAGE" in
    prepare)
        python3 "$TOOLS/prepare_build.py" --core "$CORE" --work "$WORK"
        ;;
    dependencies)
        load_build
        OFFLINE=()
        if [[ -n "${H3_WHEELHOUSE:-}" ]]; then OFFLINE=(--wheelhouse "$H3_WHEELHOUSE"); fi
        python "$TOOLS/install_dependencies.py" --work "$WORK" "${OFFLINE[@]}"
        ;;
    framework)
        load_build
        if [[ -f "$WORK/build-status.json" ]] && python -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1])).get("passed",False))' "$WORK/build-status.json"; then
            python "$TOOLS/build_framework.py" --verify-completed
        else
            RESUME=()
            if [[ -f "$WORK/build-origin.json" ]]; then RESUME=(--resume); fi
            # On lease/timeout interruption the persistent object tree remains.
            # Run this same explicit stage again in the next shared CPU session.
            timeout --signal=TERM --kill-after=20s 1650s python "$TOOLS/build_framework.py" "${RESUME[@]}"
        fi
        ;;
    ccl)
        load_build
        python -c 'import json,sys;v=json.load(open(sys.argv[1]));assert v.get("passed") and v.get("clean_host_framework")' "$WORK/build-status.json"
        timeout --signal=TERM --kill-after=20s 1650s python "$TOOLS/build_ccl.py"
        ;;
    flash)
        load_build
        FLASH_ARGS=(--output "$WORK/vendor-flash" --discover)
        if [[ -n "${H3_FLASH_SO:-}" ]]; then FLASH_ARGS+=(--flash-so "$H3_FLASH_SO"); fi
        if [[ -n "${H3_FLASH_WHEEL:-}" ]]; then FLASH_ARGS+=(--wheel "$H3_FLASH_WHEEL"); fi
        python "$TOOLS/prepare_flash.py" "${FLASH_ARGS[@]}"
        ;;
    verify)
        # Requires the allocated eight-DCU image. Does not load model weights.
        load_build
        worker_arguments
        timeout --signal=TERM --kill-after=20s 1740s python "$TOOLS/run_rebuilt.py" \
            "${WORKER_ARGS[@]}" --verify-only --output "$RESULTS/preflight"
        ;;
    communication)
        load_build
        python "$TOOLS/install_dependencies.py" --work "$WORK" --check-only
        mkdir -p "$RESULTS"
        timeout --signal=TERM --kill-after=20s 1740s python "$TOOLS/check_rebuilt_ccl.py" \
            --work "$WORK" --output "$RESULTS/communication"
        ;;
    native)
        load_build
        worker_arguments
        timeout --signal=TERM --kill-after=20s 1740s python "$TOOLS/run_rebuilt.py" \
            "${WORKER_ARGS[@]}" --backend native --output "$RESULTS/native"
        ;;
    reference)
        load_build
        worker_arguments
        timeout --signal=TERM --kill-after=20s 1740s python "$TOOLS/run_rebuilt.py" \
            "${WORKER_ARGS[@]}" --backend reference --output "$RESULTS/reference"
        ;;
    comparison-dependencies)
        load_build
        OFFLINE=()
        if [[ -n "${H3_WHEELHOUSE:-}" ]]; then OFFLINE=(--wheelhouse "$H3_WHEELHOUSE"); fi
        python "$TOOLS/install_comparison.py" --work "$WORK" "${OFFLINE[@]}"
        ;;
    compare)
        load_build
        if [[ -e "$RESULTS/compare" || -e "$RESULTS/compare.stdout.log" ]]; then
            printf '%s\n' 'Comparison output already exists; choose a new H3_OUTPUT_ROOT and explicit input videos.' >&2
            exit 1
        fi
        REFERENCE="${H3_REFERENCE_VIDEO:-$RESULTS/reference/reference-768x928-s50.mp4}"
        NATIVE="${H3_NATIVE_VIDEO:-$RESULTS/native/native-tp8-768x928-s50.mp4}"
        test -f "$REFERENCE"
        test -f "$NATIVE"
        mkdir -p "$RESULTS"
        python "$REPO/examples/minimax_h3_compare.py" --model-path "$MODEL" --tasks fl2va \
            --reference-fl2va-video "$REFERENCE" --infinilm-fl2va-video "$NATIVE" \
            --output-dir "$RESULTS/compare" | tee "$RESULTS/compare.stdout.log"
        ;;
    *)
        printf '%s\n' 'Explicit stages: prepare dependencies framework ccl flash verify communication native reference comparison-dependencies compare'
        [[ "$STAGE" == help ]]
        ;;
esac
