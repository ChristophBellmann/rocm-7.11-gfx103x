#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="$ROOT_DIR/build/dist/rocm"
HIPCC="$ROCM_PATH/bin/hipcc"
BITCODE_DIR="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
SOURCE="$ROOT_DIR/hip_gemm.cpp"
BIN="$ROOT_DIR/hip_gemm_test"
ORIGINAL_PATH="$PATH"
ORIGINAL_LD="${LD_LIBRARY_PATH:-}"
BUSY_TARGET="${1:-10}"

if ! [[ "$BUSY_TARGET" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Usage: $0 [busy_seconds]" >&2
  exit 1
fi

if [[ ! -d "$ROCM_PATH" ]]; then
  echo "ROCm tree missing at $ROCM_PATH" >&2
  exit 1
fi

if [[ ! -x "$HIPCC" ]]; then
  echo "hipcc missing at $HIPCC" >&2
  exit 1
fi

set_env() {
  export ROCM_PATH="$ROCM_PATH"
  export HIP_PATH="$ROCM_PATH"
  export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$ORIGINAL_PATH"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib${ORIGINAL_LD:+:$ORIGINAL_LD}"
}

if [[ ! -f "$BIN" || "$SOURCE" -nt "$BIN" ]]; then
  set_env
  "$HIPCC" --rocm-device-lib-path="$BITCODE_DIR" "$SOURCE" -o "$BIN"
fi

set_env

echo "Running correctness check (2048², 200 iterations)..."
START=$(date +%s.%N)
"$BIN" 2048 200
END=$(date +%s.%N)
ELAPSED_NUM=$(LC_NUMERIC=C echo "$END - $START" | bc -l)
LC_NUMERIC=C printf "HIP GEMM runtime: %.3f seconds (verification)\n" "$ELAPSED_NUM"

parse_vram() {
  "$ROCM_PATH/bin/rocm-smi" --showmeminfo vram | awk -F: '
    /VRAM Total Memory/ { total=$NF }
    /VRAM Total Used Memory/ { used=$NF }
    END {
      gsub(/[[:space:]]/, "", total)
      gsub(/[[:space:]]/, "", used)
      print used, total
    }'
}

get_vram_percent() {
  local used total
  read -r used total <<< "$(parse_vram)"
  if [[ -z "$used" || -z "$total" || "$total" -eq 0 ]]; then
    echo "0"
    return
  fi
  LC_NUMERIC=C awk -v u="$used" -v t="$total" 'BEGIN { printf "%.2f", (u * 100) / t }'
}

read -r used total <<< "$(parse_vram)"
target_fraction=0.5
target_bytes=$(LC_NUMERIC=C awk -v t="$total" -v f="$target_fraction" 'BEGIN { printf "%.0f", t * f }')
max_dimension=$(LC_NUMERIC=C awk -v b="$target_bytes" 'BEGIN { printf "%.0f", sqrt(b / 12) }')
max_dimension=$(( max_dimension < 8192 ? max_dimension : 8192 ))
(( max_dimension < 2048 )) && max_dimension=2048
target_percent=$(LC_NUMERIC=C printf "%.0f" "$(echo "$target_fraction * 100" | bc -l)")
echo "Target VRAM: $target_bytes bytes (~$target_percent%)"
echo "max N (safe) = $max_dimension"

step=512
current_N=2048

BUSY_START_NS=$(date +%s%N)
target_ns=$(LC_NUMERIC=C printf "%.0f" "$(echo "$BUSY_TARGET * 1000000000" | bc -l)")
total_end_ns=$((BUSY_START_NS + target_ns))
echo "Busy loop: increasing N every second (limit $BUSY_TARGET s)"

while (( $(date +%s%N) < total_end_ns )); do
  segment_end_ns=$(( $(date +%s%N) + 1000000000 ))
  if (( segment_end_ns > total_end_ns )); then
    segment_end_ns=$total_end_ns
  fi

  percent=$(get_vram_percent)
  LC_NUMERIC=C printf "Segment VRAM usage %.2f%%, running N=%d\n" "$percent" "$current_N"

  while (( $(date +%s%N) < segment_end_ns )); do
    "$BIN" "$current_N" 4 >/dev/null 2>&1 || true
  done

  used_percent=$(get_vram_percent)
  if (( $(LC_NUMERIC=C bc -l <<< "$used_percent >= 50") )); then
    LC_NUMERIC=C printf "VRAM target reached: %.2f%%\n" "$used_percent"
    break
  fi

  if (( current_N < max_dimension )); then
    current_N=$(( current_N + step ))
    (( current_N > max_dimension )) && current_N="$max_dimension"
  fi
done

BUSY_END_NS=$(date +%s%N)
BUSY_ELAPSED=$(LC_NUMERIC=C awk -v start="$BUSY_START_NS" -v end="$BUSY_END_NS" 'BEGIN { printf "%.3f", (end - start) / 1e9 }')
LC_NUMERIC=C printf "Additional HIP GEMM segments ran for %.3f seconds\n" "$BUSY_ELAPSED"
