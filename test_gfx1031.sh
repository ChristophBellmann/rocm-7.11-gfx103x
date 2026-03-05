#!/usr/bin/env bash
set -euo pipefail

# Ensure numeric parsing/formatting uses '.' as decimal separator regardless of
# user locale (important for printf/awk when emitting table-like output).
# Do NOT force LC_ALL=C because that breaks UTF-8 output (formulas).
export LC_NUMERIC=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ENABLED=0
LOG_FILE_DEFAULT="${ROOT}/test_gfx1031.log"
LOG_FILE="${LOG_FILE_DEFAULT}"
MODE="quick"
RUN_SANITY=1
# Default behavior is *build validation* (sanity + consistency). Benchmarks are
# opt-in because they can be slow and depend on installed bench binaries.
RUN_BENCH=0
RUN_MIOPEN=0
RUN_MIOPEN_SMOKE=0
RUN_CORE_LOAD=0
# Default: power sampling enabled (per-test 5s idle baseline + dW).
RUN_POWER=1
BUILD_DIR="${BUILD_DIR:-}"
USE_SYSTEM_ROCM=0
SYSTEM_ROCM_PATH="${SYSTEM_ROCM_PATH:-}"
RUN_CONSISTENCY=0
CONSISTENCY_DEEP=0
EXPECT_STAGE="" # "", "stage1", "stage2"
STAGE1_BUILD_DIR="${STAGE1_BUILD_DIR:-build-stage1}"
ORIG_ARGS=("$@")
USER_SELECTED_BUILD_DIR=0
BENCH_LITE=0
BENCH_MENU=0
if [[ -n "${BUILD_DIR}" ]]; then
  USER_SELECTED_BUILD_DIR=1
fi

# Output styling (TTY only).
IS_TTY=0
if [[ -t 1 ]]; then
  IS_TTY=1
fi

COLOR_ENABLED="${IS_TTY}"
if [[ -n "${NO_COLOR:-}" ]]; then
  COLOR_ENABLED=0
fi
# If logging to a file, default to plain output to keep logs readable.
if (( LOG_ENABLED )) && [[ -z "${FORCE_COLOR:-}" ]]; then
  COLOR_ENABLED=0
fi

if (( COLOR_ENABLED )); then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_CYAN=$'\033[36m'
else
  C_RESET=""
  C_BOLD=""
  C_DIM=""
  C_RED=""
  C_GREEN=""
  C_YELLOW=""
  C_CYAN=""
fi

print_formula_line() {
  local formula="$1"
  local indent="${2:-         }"

  if (( ! COLOR_ENABLED )); then
    echo "${indent}${formula}" | tee -a "${LOG_FILE}"
    return 0
  fi

  # Token-based, calm Truecolor palette (works best in dark themes).
  # - keywords/functions:  #61AFEF (97,175,239)
  # - identifiers:         #C678DD (198,120,221)
  # - operators/symbols:   #E5C07B (229,192,123)
  # - numbers/units:       #D19A66 (209,154,102)
  # - brackets/indices:    #7F848E (127,132,142)
  # - variables A/B/C:     per-variable colors (distinct)
  # IMPORTANT: Keep the tokenizer ASCII-only. Treat any non-ASCII bytes as
  # opaque to avoid corrupting UTF-8 sequences (which would render as �).
  printf '%s\n' "${formula}" | awk '
    BEGIN{
      RST="\033[0m"
      KW="\033[38;2;97;175;239m"
      ID="\033[38;2;198;120;221m"
      OP="\033[38;2;229;192;123m"
      NUM="\033[38;2;209;154;102m"
      BR="\033[38;2;127;132;142m"
      VA="\033[38;2;224;108;117m"     # #E06C75 (A)
      VB="\033[38;2;86;182;194m"      # #56B6C2 (B)
      VC="\033[38;2;152;195;121m"     # #98C379 (C)
      VQ="\033[38;2;209;154;102m"     # #D19A66 (Q/R/L/U/P/x/y)
      VV="\033[38;2;229;192;123m"     # #E5C07B (m/n/k/N/nnz/iters/batch/...)
      # simple keyword set
      kw["GEMM"]=1; kw["FFT"]=1; kw["QR"]=1; kw["LU"]=1; kw["RNG"]=1; kw["AXP"]=1
      kw["factorization"]=1; kw["forward"]=1; kw["batched"]=1; kw["Philox"]=1; kw["engine"]=1
      kw["sizeof"]=1; kw["log2"]=1
    }
    function isop(c){ return index("=+-*/^,;:", c) > 0 }
    function isbr(c){ return index("()[]{}", c) > 0 }
    function isnum(c){ return c ~ /[0-9.]/ }
    function isword(c){ return c ~ /[A-Za-z0-9_-]/ }
    function emit(col, tok){ printf("%s%s%s", col, tok, RST) }
    function var_color(tok){
      if(tok=="A") return VA
      if(tok=="B") return VB
      if(tok=="C") return VC
      if(tok=="Q"||tok=="R"||tok=="L"||tok=="U"||tok=="P") return VQ
      if(tok=="x"||tok=="y") return VQ
      if(tok=="m"||tok=="n"||tok=="k"||tok=="N"||tok=="i"||tok=="j") return VV
      if(tok=="nnz"||tok=="batch"||tok=="iters"||tok=="count"||tok=="nnz_eff") return VV
      return ""
    }
    {
      s=$0
      i=1
      n=length(s)
      while(i<=n){
        c=substr(s,i,1)
        # Non-ASCII byte => part of UTF-8 sequence, print as-is.
        # Use octal escapes (portable across awk variants; avoids \xNN pitfalls).
        if (c !~ /^[\001-\177]$/) { printf("%s", c); i++; continue }
        if(c ~ /[[:space:]]/){ printf("%s", c); i++; continue }
        if(isbr(c)){ emit(BR, c); i++; continue }
        if(isop(c)){ emit(OP, c); i++; continue }
        # numbers/units
        if(isnum(c)){
          tok=c
          i++
          while(i<=n && isnum(substr(s,i,1))){ tok=tok substr(s,i,1); i++ }
          emit(NUM, tok); continue
        }
        # words/identifiers
        if(isword(c)){
          tok=c
          i++
          while(i<=n && isword(substr(s,i,1))){ tok=tok substr(s,i,1); i++ }
          vc=var_color(tok)
          if(vc!=""){ emit(vc, tok) }
          else if(tok in kw){ emit(KW, tok) }
          else { emit(ID, tok) }
          continue
        }
        # fallback
        printf("%s", c); i++
      }
      printf("\n")
    }' | sed "s/^/${indent}/" | tee -a "${LOG_FILE}"
}

print_bench_header() {
  local label="$1"
  local expected="$2"
  local w=44
  # Ensure a blank line between bench blocks for scanability.
  if [[ -n "${BENCH_HEADER_COUNT:-}" && "${BENCH_HEADER_COUNT}" -gt 0 ]]; then
    echo "" | tee -a "${LOG_FILE}"
  fi
  BENCH_HEADER_COUNT=$(( ${BENCH_HEADER_COUNT:-0} + 1 ))
  # One-line, dominant header: marker, bold bench name, dim expected.
  printf "%s==>%s %s%-*s%s %s(expected: %s)%s\n" \
    "${C_CYAN}" "${C_RESET}" "${C_BOLD}" "${w}" "$(fmt_label "${label}")" "${C_RESET}" \
    "${C_DIM}" "${expected}" "${C_RESET}" | tee -a "${LOG_FILE}"
}

print_bench_anchor() {
  local anchor="$1"
  printf "    %s%s:%s\n" "${C_CYAN}" "${anchor}" "${C_RESET}" | tee -a "${LOG_FILE}"
}

print_bench_desc() {
  # args: desc [indent] [style]
  # style: plain|dim
  local desc="$1"
  local indent="${2:-         }"
  local style="${3:-plain}"
  if [[ -z "${desc}" ]]; then
    return 0
  fi
  local prefix=""
  if [[ "${style}" == "dim" ]]; then
    prefix="${C_DIM}"
  fi
  printf "%s%s%s%s\n" "${indent}" "${prefix}" "${desc}" "${C_RESET}" | tee -a "${LOG_FILE}"
}

print_colorized_text() {
  # args: text [indent] [style]
  # Token-based colorization using the same palette as formulas, extended with
  # per-variable colors for A/B/C/Q/R/L/U/P/x/y and common size symbols.
  local text="$1"
  local indent="${2:-         }"
  local style="${3:-plain}"

  if (( ! COLOR_ENABLED )); then
    print_bench_desc "${text}" "${indent}" "${style}"
    return 0
  fi

  local dim_prefix=""
  if [[ "${style}" == "dim" ]]; then
    dim_prefix="${C_DIM}"
  fi

  printf '%s\n' "${text}" | awk -v dim="${dim_prefix}" '
    BEGIN{
      RST="\033[0m"
      KW="\033[38;2;97;175;239m"         # #61AFEF (reserve for math-ish keywords)
      OP="\033[38;2;229;192;123m"        # #E5C07B
      NUM="\033[38;2;209;154;102m"       # #D19A66
      BR="\033[38;2;127;132;142m"        # #7F848E
      VA="\033[38;2;224;108;117m"        # #E06C75 (A)
      VB="\033[38;2;86;182;194m"         # #56B6C2 (B)
      VC="\033[38;2;152;195;121m"        # #98C379 (C)
      VQ="\033[38;2;209;154;102m"        # #D19A66 (Q/R/L/U/P)
      VV="\033[38;2;229;192;123m"        # #E5C07B (m/n/k/N/nnz/batch/iters/count)

      # Keep coloring focused on math symbols/variables. Only a few "math-ish"
      # words are highlighted to help scanning.
      kw["ops_FLOP"]=1; kw["data_B"]=1; kw["ops_samples"]=1
      kw["sizeof"]=1; kw["log2"]=1
    }
    function isop(c){ return index("=+-*/^,;:<>", c) > 0 }
    function isbr(c){ return index("()[]{}", c) > 0 }
    function isnum(c){ return c ~ /[0-9.]/ }
    function isword(c){ return c ~ /[A-Za-z0-9_]/ }
    function emit(col, tok){ printf("%s%s%s", col, tok, RST) }
    function emit_dim_plain(tok){ printf("%s%s%s", dim, tok, RST) }
    function emit_dim(col, tok){ printf("%s%s%s%s", dim, col, tok, RST) }
    function emit_any(col, tok){ if(dim!=""){ emit_dim(col, tok) } else { emit(col, tok) } }
    function var_color(tok){
      if(tok=="A") return VA
      if(tok=="B") return VB
      if(tok=="C") return VC
      if(tok=="Q"||tok=="R"||tok=="L"||tok=="U"||tok=="P") return VQ
      if(tok=="x"||tok=="y") return VQ
      if(tok=="m"||tok=="n"||tok=="k"||tok=="N"||tok=="i"||tok=="j") return VV
      if(tok=="nnz"||tok=="batch"||tok=="iters"||tok=="count"||tok=="nnz_eff") return VV
      return ""
    }
    {
      s=$0
      i=1
      n=length(s)
      while(i<=n){
        c=substr(s,i,1)
        # Non-ASCII byte => part of UTF-8 sequence, print as-is.
        if (c !~ /^[\001-\177]$/) { printf("%s", c); i++; continue }
        if(c ~ /[[:space:]]/){ printf("%s", c); i++; continue }
        if(isbr(c)){ emit_any(BR, c); i++; continue }
        if(isop(c)){ emit_any(OP, c); i++; continue }
        if(isnum(c)){
          tok=c
          i++
          while(i<=n && isnum(substr(s,i,1))){ tok=tok substr(s,i,1); i++ }
          emit_any(NUM, tok); continue
        }
        if(isword(c)){
          tok=c
          i++
          while(i<=n && isword(substr(s,i,1))){ tok=tok substr(s,i,1); i++ }
          vc=var_color(tok)
          if(vc!=""){ emit_any(vc, tok); continue }
          if(tok in kw){ emit_any(KW, tok); continue }
          # Default: do not color normal words; keep emphasis on symbols/vars.
          if(dim!=""){ emit_dim_plain(tok) } else { printf("%s", tok) }
          continue
        }
        printf("%s", c); i++
      }
      printf("\n")
    }' | sed "s/^/${indent}/" | tee -a "${LOG_FILE}"
}

LAST_MODEL_KIND="${LAST_MODEL_KIND:-}"

print_ops_data_model() {
  # Emits a short, concrete mathematical model for ops/data based on the bench kind.
  # This is *not* a performance claim; it is a parameter-based accounting sketch.
  local kind="$1"   # GEMM|QR|LU|AXPYI|FFT|RNG|HIPLOAD|CONV
  local do_full=1
  if [[ -n "${LAST_MODEL_KIND}" && "${LAST_MODEL_KIND}" == "${kind}" ]]; then
    do_full=0
  fi

  if (( do_full == 0 )); then
    print_colorized_text "- ops/data model: same as the previous ${kind} benchmark above." "         " "dim"
    return 0
  fi

  print_colorized_text "- Notation: ops = algorithmic operation count of the kernel math (model), not a hardware counter."
  print_colorized_text "- Notation: data = logical tensor I/O bytes touched (reads+writes) by that math model, not measured DRAM traffic."

  case "${kind}" in
    GEMM)
      print_colorized_text "- Interpretation: each output element C[i,j] is a length-k dot product, then scaled (α) and accumulated with the prior C via β."
      print_colorized_text "- ops_FLOP ≈ iters · 2·m·n·k (multiply+add) (optionally + iters·2·m·n for β·C + …; usually negligible)"
      print_colorized_text "- data_B ≈ iters · (sizeof(A)·m·k + sizeof(B)·k·n + 2·sizeof(C)·m·n) (C is read+written once per update; cache/reuse effects excluded)"
      ;;
    QR)
      print_colorized_text "- Interpretation: for each matrix in the batch, compute A=Q·R with Qᵀ·Q=I (m×n, m≥n), repeated iters times."
      print_colorized_text "- ops_FLOP ≈ batch · iters · (2·m·n² − (2/3)·n³) (for m≥n, Householder-QR; rough)"
      print_colorized_text "- data_B ≈ batch · (sizeof(A)·m·n + sizeof(tau)·n) (+ workspace and panel traffic, implementation-dependent)"
      ;;
    LU)
      print_colorized_text "- Interpretation: factor A with partial pivoting into P·A=L·U (P is a permutation), repeated iters times."
      print_colorized_text "- ops_FLOP ≈ iters · (2/3)·n³ (for n×n)"
      print_colorized_text "- data_B ≈ iters · sizeof(A)·n² (+ pivot vector/workspace traffic)"
      ;;
    AXPYI)
      print_colorized_text "- Interpretation: stream nnz indexed updates y[i_j] += α·x_j; nnz_eff reflects how many distinct y entries are touched."
      print_colorized_text "- ops_FLOP ≈ iters · 2·nnz (multiply+add)"
      print_colorized_text "- data_B ≈ iters · (sizeof(x)·nnz + sizeof(i)·nnz + 2·sizeof(y)·nnz_eff) (y is read+written; nnz_eff depends on index repeats)"
      ;;
    FFT)
      print_colorized_text "- Interpretation: compute batched N-point forward DFTs; complexity is Θ(N·log2(N)) per transform (constant depends on the plan/kernels)."
      print_colorized_text "- ops ≈ iters · batch · c·N·log2(N) (constant c is implementation-dependent)"
      print_colorized_text "- data_B ≈ iters · batch · sizeof(complex)·N·(reads+writes) (typically ≈2)"
      ;;
    RNG)
      print_colorized_text "- Interpretation: generate count independent, identically distributed samples xᵢ ∼ U(0,1) and write them out, repeated iters times."
      print_colorized_text "- ops_samples = iters · count"
      print_colorized_text "- data_B ≈ iters · count · sizeof(output)"
      ;;
    HIPLOAD)
      print_colorized_text "- Interpretation: launch a native HIP kernel repeatedly; each thread performs inner fused multiply-add updates on its element and writes the result back."
      print_colorized_text "- ops_FLOP ≈ reps · N · 2·inner (dominant FMA term; scalar update overhead omitted)"
      print_colorized_text "- data_B ≈ reps · N · (read A + read B + read/write C) ≈ reps · N · (4+4+8) B"
      ;;
    CONV)
      print_colorized_text "- Interpretation: run repeated forward 2D convolutions (NCHW) to sustain MIOpen kernel execution and runtime scheduling."
      print_colorized_text "- ops_FLOP ≈ reps · 2·N·H_out·W_out·K·C·Y·X (multiply+add MAC accounting)"
      print_colorized_text "- data_B ≈ reps · (sizeof(X)·N·C·H·W + sizeof(W)·K·C·Y·X + sizeof(Y)·N·K·H_out·W_out)"
      ;;
  esac

  LAST_MODEL_KIND="${kind}"
}

fmt_sci() {
  # args: number -> "m.e" scientific notation (one decimal place), e.g. 1.7e11
  local value="$1"
  awk -v v="${value}" '
    BEGIN{
      v=v+0.0
      if(v==0.0){ print "0"; exit }
      a=(v<0?-v:v)
      e=int(log(a)/log(10))
      m=v/(10^e)
      # normalize mantissa to [1,10)
      while(m>=10){ m/=10; e++ }
      while(m<1){ m*=10; e-- }
      printf("%.1fe%d\n", m, e)
    }'
}

set_bench_meta_stats() {
  # args: ops ops_unit bytes
  local ops="$1"
  local ops_unit="$2"
  local bytes="$3"
  printf "ops ≈ %s %s;  data ≈ %s B" "$(fmt_sci "${ops}")" "${ops_unit}" "$(fmt_sci "${bytes}")"
}

status_color() {
  local status="$1"
  if (( ! COLOR_ENABLED )); then
    echo ""
    return 0
  fi
  case "${status}" in
    OK) echo "${C_GREEN}" ;;
    FAIL) echo "${C_RED}" ;;
    SKIP) echo "${C_YELLOW}" ;;
    *) echo "" ;;
  esac
}

usage() {
  cat <<'EOF_USAGE'
Usage: test_gfx1031.sh [options]

Default behavior:
  - If invoked with no args in an interactive terminal (TTY): opens the interactive bench menu (1–12).
  - Otherwise: runs sanity checks only (no benchmarks).
  - ROCm source defaults to in-tree build output (`<repo>/<build-dir>/dist/rocm`).
    System ROCm is only used with `--system-rocm` / `--system-rocm-path`.
  - Power sampling is ON by default (5s idle baseline + per-test energy/utilization).

Options:
  --quick        Select quick benchmark sizes (default mode)
  --full         Select longer benchmark sizes (bigger sizes / more iters)
  --power        Sample GPU power/utilization via sysfs during sanity/bench (adds baseline + per-test metrics)
  --no-power     Disable power sampling (override default)
  --bench        Run performance benchmarks (in addition to sanity)
  --bench-lite   Run only the lightweight BLAS GEMM benchmarks (rocBLAS + hipBLAS)
  --bench-menu   Interactive bench menu (select 1-12; 0=all; q=quit)
  --log [file]   Enable logging to file (default: test_gfx1031.log)
  --no-bench     Skip performance benchmarks (default)
  --bench-only   Run benchmarks only (no sanity)
  --miopen       Check MIOpen + composable_kernel artifacts (and MIOpenDriver --version if present)
  --miopen-smoke Run a tiny MIOpenDriver smoke test (may take time on first run)
  --core-load    Run lightweight component load tests (HIP/HSA, rocBLAS, MIOpen)
  --no-core-load Disable component load tests
  --consistency  Run build/toolchain consistency checks
  --consistency-only
                Run consistency checks only
  --deep         Deep consistency checks (scan more files)
  --expect-stage1
                Expect Stage-1 (system clang allowed)
  --expect-stage2
                Expect Stage-2 (no system clang/llvm-18 fallback)
  --stage1       Use BUILD_DIR=build-stage1
  --stage2       Use BUILD_DIR=build-stage2
  --build-dir <dir>
                Override build directory (default: auto; prefers build-stage2, then build, then build-stage1)
  --system-rocm
                Use system ROCm instead of in-tree build ROCm (default system path: /opt/rocm)
  --system-rocm-path <dir>
                Use system ROCm from explicit path (implies --system-rocm)
  -h, --help     Show this help

Environment overrides:
  BENCH_SIZE       override GEMM size (default 2048 quick, 4096 full)
  BENCH_ITERS      override iterations (default 10 quick, 20 full)
  TEST_LOG         override log file (only used if --log is set)
  BUILD_DIR        force a specific build directory (disables auto-multi-builddir loop)
  ROCM_PATH        ignored by default; only used when --system-rocm is selected
  SYSTEM_ROCM_PATH default path for --system-rocm (if --system-rocm-path is not set)
  TEST_GFX1031_SINGLE
                  set to 1 to disable auto-multi-builddir loop
  TEST_SKIP_VENV   set to 1 to skip activating .venv (useful in containers)
  NO_COLOR         disable colored output
  FORCE_COLOR      force colored output even when logging
  STAGE1_BUILD_DIR Stage-1 build dir for Stage-2 expectations (default: build-stage1)
EOF_USAGE
}

choose_default_build_dir() {
  # If user set BUILD_DIR explicitly (env or --build-dir), keep it.
  if [[ -n "${BUILD_DIR:-}" ]]; then
    return 0
  fi
  # Prefer Stage-2 if present (most useful for users).
  if [[ -d "${ROOT}/build-stage2/dist/rocm" ]]; then
    BUILD_DIR="build-stage2"
    return 0
  fi
  # Fall back to a default in-tree build dir.
  if [[ -d "${ROOT}/build/dist/rocm" ]]; then
    BUILD_DIR="build"
    return 0
  fi
  # Finally Stage-1 toolchain dist (toolchain-only; limited runtime tools).
  if [[ -d "${ROOT}/build-stage1/dist/rocm" ]]; then
    BUILD_DIR="build-stage1"
    return 0
  fi
  # As a last resort, keep the traditional "build" name so error messages are stable.
  BUILD_DIR="build"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      MODE="quick"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --power)
      RUN_POWER=1
      shift
      ;;
    --no-power)
      RUN_POWER=0
      shift
      ;;
    --bench)
      RUN_BENCH=1
      RUN_CORE_LOAD=1
      shift
      ;;
    --bench-lite)
      RUN_BENCH=1
      BENCH_LITE=1
      shift
      ;;
    --bench-menu)
      RUN_SANITY=0
      RUN_BENCH=1
      BENCH_MENU=1
      shift
      ;;
    --log)
      LOG_ENABLED=1
      if [[ -n "${2:-}" && "${2:-}" != --* ]]; then
        LOG_FILE="$2"
        shift 2
      else
        shift
      fi
      ;;
    --no-bench)
      RUN_BENCH=0
      shift
      ;;
    --bench-only)
      RUN_SANITY=0
      RUN_BENCH=1
      shift
      ;;
    --miopen)
      RUN_MIOPEN=1
      shift
      ;;
    --miopen-smoke)
      RUN_MIOPEN=1
      RUN_MIOPEN_SMOKE=1
      shift
      ;;
    --core-load)
      RUN_CORE_LOAD=1
      shift
      ;;
    --no-core-load)
      RUN_CORE_LOAD=0
      shift
      ;;
    --consistency)
      RUN_CONSISTENCY=1
      shift
      ;;
    --consistency-only)
      RUN_SANITY=0
      RUN_BENCH=0
      RUN_CONSISTENCY=1
      shift
      ;;
    --deep)
      CONSISTENCY_DEEP=1
      shift
      ;;
    --expect-stage1)
      EXPECT_STAGE="stage1"
      shift
      ;;
    --expect-stage2)
      EXPECT_STAGE="stage2"
      shift
      ;;
    --stage1)
      BUILD_DIR="build-stage1"
      USER_SELECTED_BUILD_DIR=1
      shift
      ;;
    --stage2)
      BUILD_DIR="build-stage2"
      USER_SELECTED_BUILD_DIR=1
      shift
      ;;
    --build-dir)
      BUILD_DIR="${2:-}"
      USER_SELECTED_BUILD_DIR=1
      shift 2
      ;;
    --system-rocm)
      USE_SYSTEM_ROCM=1
      shift
      ;;
    --system-rocm-path)
      USE_SYSTEM_ROCM=1
      SYSTEM_ROCM_PATH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Default UX: if invoked without args in an interactive terminal, open the bench menu.
if (( ${#ORIG_ARGS[@]} == 0 )) && [[ -t 0 ]]; then
  RUN_SANITY=0
  RUN_BENCH=1
  BENCH_MENU=1
fi

# If no build dir was specified, and we have multiple in-tree dist roots,
# run the same tests for each build dir automatically (stage2/build/stage1).
if [[ -z "${TEST_GFX1031_SINGLE:-}" && ${USER_SELECTED_BUILD_DIR} -eq 0 ]]; then
  # If the user sets an explicit expectation, interpret it as a selection.
  if [[ "${EXPECT_STAGE}" == "stage2" ]]; then
    BUILD_DIR="build-stage2"
    USER_SELECTED_BUILD_DIR=1
  elif [[ "${EXPECT_STAGE}" == "stage1" ]]; then
    BUILD_DIR="build-stage1"
    USER_SELECTED_BUILD_DIR=1
  else
    # For benchmarks, default to Stage-2 if available (Stage-1 is typically
    # toolchain-only and does not ship rocblas-bench/hipblas-bench).
    if (( RUN_BENCH )) && [[ -d "${ROOT}/build-stage2/dist/rocm" ]]; then
      BUILD_DIR="build-stage2"
      USER_SELECTED_BUILD_DIR=1
    fi

    if (( USER_SELECTED_BUILD_DIR == 0 )); then
      build_dirs=()
      for d in build-stage2 build build-stage1; do
        if [[ -d "${ROOT}/${d}/dist/rocm" ]]; then
          build_dirs+=("${d}")
        fi
      done
      if (( ${#build_dirs[@]} > 1 )); then
        overall_rc=0
        for d in "${build_dirs[@]}"; do
          echo "==== build dir: ${d} ===="
        if (( LOG_ENABLED )); then
          base_log="${LOG_FILE_DEFAULT}"
          if [[ "${base_log}" == *.log ]]; then
            this_log="${base_log%.log}.${d}.log"
          else
            this_log="${base_log}.${d}.log"
          fi
          TEST_GFX1031_SINGLE=1 BUILD_DIR="${d}" "${ROOT}/test_gfx1031.sh" --log "${this_log}" "${ORIG_ARGS[@]}" || overall_rc=1
        else
          TEST_GFX1031_SINGLE=1 BUILD_DIR="${d}" "${ROOT}/test_gfx1031.sh" "${ORIG_ARGS[@]}" || overall_rc=1
        fi
      done
      exit "${overall_rc}"
    fi
  fi
fi
fi

choose_default_build_dir

if (( LOG_ENABLED )); then
  if [[ -n "${TEST_LOG:-}" ]]; then
    LOG_FILE="${TEST_LOG}"
  fi
else
  # Default: no log file is created. We still pipe through a log sink for
  # simplicity (tee -a /dev/null).
  LOG_FILE="/dev/null"
fi

IN_CONTAINER=0
if [[ -f "/.dockerenv" ]]; then
  IN_CONTAINER=1
else
  if [[ -r "/proc/1/cgroup" ]] && grep -qE '(docker|containerd|kubepods)' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=1
  fi
fi

# Some workflows mount the repo into a container to validate Stage-2 dist artifacts.
# In that case, a host-created venv can be incompatible with the container userland.
# Default: skip venv activation in containers unless explicitly forced.
SKIP_VENV=0
if [[ "${TEST_SKIP_VENV:-}" == "1" ]] || [[ "${SKIP_VENV:-}" == "1" ]]; then
  SKIP_VENV=1
fi
if (( IN_CONTAINER )) && [[ -z "${THEROCK_FORCE_VENV:-}" ]]; then
  SKIP_VENV=1
fi

if (( ! SKIP_VENV )) && [[ -f "${ROOT}/.venv/bin/activate" ]] && [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # Activate venv for helper tools if present.
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

if (( LOG_ENABLED )); then
  if [[ -f "${LOG_FILE}" ]]; then
    ts="$(date +%Y%m%d-%H%M%S)"
    mv "${LOG_FILE}" "${LOG_FILE}.bak-${ts}"
  fi
  touch "${LOG_FILE}"
fi

ROCM_PATH_DEFAULT="${ROOT}/${BUILD_DIR}/dist/rocm"
if (( USE_SYSTEM_ROCM )); then
  if [[ -n "${SYSTEM_ROCM_PATH}" ]]; then
    ROCM_PATH="${SYSTEM_ROCM_PATH}"
  elif [[ -n "${ROCM_PATH:-}" ]]; then
    ROCM_PATH="${ROCM_PATH}"
  else
    ROCM_PATH="/opt/rocm"
  fi
else
  ROCM_PATH="${ROCM_PATH_DEFAULT}"
fi
HAVE_ROCM_ENV=0
if [[ -d "${ROCM_PATH}" ]]; then
  HAVE_ROCM_ENV=1
  export ROCM_PATH
  export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
  export HSA_PATH="${HSA_PATH:-$ROCM_PATH}"
  export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:${PATH:-}"
  # Include host BLAS in-tree prefix for benchmarks (lib/host-math/lib).
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/lib/host-math/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}"
  # Some packaging flows may drop OpenBLAS SONAME symlinks. For benchmark runs,
  # provide a local fallback without mutating the in-tree dist.
  _host_blas_lib="$ROCM_PATH/lib/host-math/lib"
  if [[ -d "${_host_blas_lib}" ]] && [[ ! -e "${_host_blas_lib}/librocm-openblas.so.0" ]] && [[ -e "${_host_blas_lib}/librocm-openblas.so.0.3" ]]; then
    _tmp_blas="$(mktemp -d)"
    ln -s "${_host_blas_lib}/librocm-openblas.so.0.3" "${_tmp_blas}/librocm-openblas.so.0"
    ln -s "${_host_blas_lib}/librocm-openblas.so.0.3" "${_tmp_blas}/librocm-openblas.so"
    export LD_LIBRARY_PATH="${_tmp_blas}:${LD_LIBRARY_PATH}"
  fi
  if [[ -z "${HIP_DEVICE_LIB_PATH:-}" ]]; then
    if [[ -d "$ROCM_PATH/lib/llvm/amdgcn/bitcode" ]]; then
      export HIP_DEVICE_LIB_PATH="$ROCM_PATH/lib/llvm/amdgcn/bitcode"
    else
      export HIP_DEVICE_LIB_PATH="$ROCM_PATH/amdgcn/bitcode"
    fi
  fi
  if (( USE_SYSTEM_ROCM )); then
    echo "${C_GREEN}Activated system ROCm:${C_RESET} ${ROCM_PATH}" | tee -a "${LOG_FILE}"
  else
    echo "${C_GREEN}Activated in-tree ROCm:${C_RESET} ${ROCM_PATH}" | tee -a "${LOG_FILE}"
  fi
else
  if (( RUN_SANITY )) || (( RUN_BENCH )); then
    echo "ROCM_PATH not found: ${ROCM_PATH}" | tee -a "${LOG_FILE}" >&2
    echo "Build first (expected default: ${ROCM_PATH_DEFAULT})." | tee -a "${LOG_FILE}" >&2
    exit 1
  fi
  echo "ROCM_PATH not found: ${ROCM_PATH} (OK for --consistency-only; skipping runtime/HIP checks)" | tee -a "${LOG_FILE}"
fi

RESULT_LABELS=()
RESULT_STATUS=()
RESULT_TIME=()
RESULT_METRIC=()

add_result() {
  RESULT_LABELS+=("$1")
  RESULT_STATUS+=("$2")
  RESULT_TIME+=("$3")
  RESULT_METRIC+=("$4")
}

time_to_seconds() {
  local t="$1"
  if [[ "${t}" == *ms ]]; then
    local ms="${t%ms}"
    awk -v v="${ms}" 'BEGIN{printf "%.3f", v/1000.0}'
    return 0
  fi
  if [[ "${t}" == *s ]]; then
    local s="${t%s}"
    # Some callers may pass "0s" or already have decimals.
    awk -v v="${s}" 'BEGIN{printf "%.3f", v+0}'
    return 0
  fi
  # Fallback: unknown format.
  awk 'BEGIN{printf "%.3f", 0.0}'
}

split_metric() {
  # args: metric -> prints "perf<TAB>power"
  local metric="$1"
  if [[ "${metric}" == *" | "* ]]; then
    local perf="${metric%% | *}"
    local power="${metric#* | }"
    printf "%s\t%s" "${perf}" "${power}"
  else
    printf "%s\t" "${metric}"
  fi
}

normalize_perf() {
  local perf="$1"
  # Prefer a compact "KEY value" style.
  perf="${perf//= / }"
  perf="${perf//=/ }"
  # Some metrics are empty.
  echo "${perf}"
}

extract_power_field() {
  # args: power_blob key_regex -> value
  local blob="$1"
  local key="$2"
  echo "${blob}" | sed -nE "s/.*${key}= *([^ ]+).*/\\1/p" | head -n 1
}

print_summary_table() {
  local title="$1"
  echo "${title}" | tee -a "${LOG_FILE}"
  printf "ID  %-30s  %-4s  %7s  %-28s\n" "BENCH" "ST" "TIME" "PERF" | tee -a "${LOG_FILE}"
  printf "%s\n" "----------------------------------------------------------------------------------------------------" | tee -a "${LOG_FILE}"

  for i in "${!RESULT_LABELS[@]}"; do
    local label="${RESULT_LABELS[$i]}"
    local status="${RESULT_STATUS[$i]}"
    local time="${RESULT_TIME[$i]}"
    local metric="${RESULT_METRIC[$i]}"

    local id=$((i + 1))
    local name="${label}"
    if [[ "${name}" == bench:* ]]; then
      name="${name#bench: }"
    fi

    local seconds
    seconds="$(time_to_seconds "${time}")"

    local perf="" power=""
    if [[ -n "${metric}" ]]; then
      if [[ "${metric}" == *" | "* ]]; then
        perf="${metric%% | *}"
        power="${metric#* | }"
      else
        perf="${metric}"
        power=""
      fi
      perf="$(normalize_perf "${perf}")"
    fi

    local st_c
    st_c="$(status_color "${status}")"
    # Print in segments so ANSI escapes don't affect column alignment.
    printf "%02d  " "${id}" | tee -a "${LOG_FILE}"
    printf "%s%-30.30s%s" "${C_BOLD}" "${name}" "${C_RESET}" | tee -a "${LOG_FILE}"
    printf "  " | tee -a "${LOG_FILE}"
    printf "%s%-4s%s" "${st_c}" "${status}" "${C_RESET}" | tee -a "${LOG_FILE}"
    printf "  %6.3fs  %-28.28s\n" "${seconds}" "${perf}" | tee -a "${LOG_FILE}"

    # Bench rows get a second line with energy/utilization fields (if power is enabled and available).
    if (( RUN_POWER )) && [[ -n "${power}" ]] && [[ "${label}" == bench:* || "${label}" == *" load"* ]]; then
      local e="" avgw="" maxw="" dw="" gpu="" mem=""
      if [[ -n "${power}" ]]; then
        e="$(extract_power_field "${power}" "E")"
        avgw="$(extract_power_field "${power}" "avgW")"
        maxw="$(extract_power_field "${power}" "maxW")"
        dw="$(extract_power_field "${power}" "dW")"
        gpu="$(extract_power_field "${power}" "gpu%")"
        mem="$(extract_power_field "${power}" "mem%")"
      fi

      [[ -z "${e}" ]] && e="n/a"
      [[ -z "${avgw}" ]] && avgw="n/a"
      [[ -z "${maxw}" ]] && maxw="n/a"
      [[ -z "${dw}" ]] && dw="n/a"
      [[ -z "${gpu}" ]] && gpu="n/a"
      [[ -z "${mem}" ]] && mem="n/a"

      printf "    Energy: %-8s  avg %-6s  max %-6s  ΔW %-7s  gpu %3s%%  mem %3s%%\n" \
        "${e}" "${avgw}" "${maxw}" "${dw}" "${gpu}" "${mem}" | tee -a "${LOG_FILE}"
    fi
    # One blank line per ID block for scanability.
    echo "" | tee -a "${LOG_FILE}"
  done
}

miopen_find_driver() {
  if command -v MIOpenDriver >/dev/null 2>&1; then
    echo "MIOpenDriver"
    return 0
  fi
  if command -v miopen-driver >/dev/null 2>&1; then
    echo "miopen-driver"
    return 0
  fi
  return 1
}

check_miopen_artifacts() {
  local label_prefix="$1"
  local start=$SECONDS
  local ok=0

  # MIOpen library presence
  if [[ -e "${ROCM_PATH}/lib/libMIOpen.so" || -n "$(ls -1 "${ROCM_PATH}/lib/libMIOpen.so"* 2>/dev/null | head -n 1)" ]]; then
    add_result "${label_prefix} miopen library" "OK" "0s" "libMIOpen found"
    ok=1
  elif [[ -e "${ROCM_PATH}/lib64/libMIOpen.so" || -n "$(ls -1 "${ROCM_PATH}/lib64/libMIOpen.so"* 2>/dev/null | head -n 1)" ]]; then
    add_result "${label_prefix} miopen library" "OK" "0s" "libMIOpen found (lib64)"
    ok=1
  else
    add_result "${label_prefix} miopen library" "SKIP" "0s" "libMIOpen not found under ${ROCM_PATH}/lib{,64} (may be disabled)"
  fi

  # composable_kernel headers (may or may not be installed depending on packaging)
  if [[ -d "${ROCM_PATH}/include/ck" || -d "${ROCM_PATH}/include/composable_kernel" ]]; then
    add_result "${label_prefix} ck headers" "OK" "0s" "headers present"
  else
    add_result "${label_prefix} ck headers" "SKIP" "0s" "not found under ${ROCM_PATH}/include (may be ok)"
  fi

  local elapsed=$((SECONDS - start))
  if (( ok )); then
    add_result "${label_prefix} miopen artifacts" "OK" "${elapsed}s" ""
  else
    add_result "${label_prefix} miopen artifacts" "SKIP" "${elapsed}s" "MIOpen not installed in this dist"
  fi
}

run_miopen_checks() {
  local label_prefix="$1"
  if (( HAVE_ROCM_ENV == 0 )); then
    add_result "${label_prefix} miopen" "FAIL" "0s" "ROCM_PATH missing"
    return 1
  fi
  check_miopen_artifacts "${label_prefix}" || true

  local drv
  if drv="$(miopen_find_driver)"; then
    run_timed "${label_prefix} driver --version" "<5s" "${drv}" --version || true
    if (( RUN_MIOPEN_SMOKE )); then
      # Very small conv; kernel compilation may still take time on first run.
      # Use conservative sizes to keep it quick if cache is warm.
      run_timed "${label_prefix} conv (smoke)" "30-180s" "${drv}" conv -n 1 -c 1 -H 8 -W 8 -k 1 -y 3 -x 3 -p 1 -q 1 || true
    else
      add_result "${label_prefix} conv (smoke)" "SKIP" "0s" "use --miopen-smoke to run"
    fi
  else
    add_result "${label_prefix} driver --version" "SKIP" "0s" "MIOpenDriver/miopen-driver not in PATH"
    add_result "${label_prefix} conv (smoke)" "SKIP" "0s" "MIOpenDriver/miopen-driver not in PATH"
  fi
}

detect_expect_stage() {
  if [[ -n "${EXPECT_STAGE}" ]]; then
    return 0
  fi
  case "${BUILD_DIR}" in
    *stage2*) EXPECT_STAGE="stage2" ;;
    *stage1*) EXPECT_STAGE="stage1" ;;
    *) EXPECT_STAGE="" ;;
  esac
}

check_cmd_available() {
  local label="$1"
  local exe="$2"
  if command -v "${exe}" >/dev/null 2>&1; then
    add_result "${label}" "OK" "0s" "$(command -v "${exe}")"
    return 0
  fi
  add_result "${label}" "FAIL" "0s" "missing: ${exe}"
  return 1
}

check_no_matches_in_files() {
  local label="$1"
  local expected="$2"
  local pattern="$3"
  shift 3
  local -a files=("$@")
  local tmp
  tmp="$(mktemp)"
  local start=$SECONDS
  set +e
  if command -v rg >/dev/null 2>&1; then
    rg -nH "${pattern}" "${files[@]}" >"${tmp}" 2>/dev/null
  else
    grep -nH -E "${pattern}" "${files[@]}" >"${tmp}" 2>/dev/null
  fi
  local rc=$?
  set -e
  local elapsed=$((SECONDS - start))
  if [[ ${rc} -eq 0 ]]; then
    local sample
    sample="$(head -n 3 "${tmp}" | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
    add_result "${label}" "FAIL" "${elapsed}s" "matched (${expected}): ${sample}"
    rm -f "${tmp}"
    return 1
  fi
  add_result "${label}" "OK" "${elapsed}s" "${expected}"
  rm -f "${tmp}"
  return 0
}

check_no_opt_rocm_in_caches() {
  local label="$1"
  local expected="$2"
  local filter="${3:-}"
  local start=$SECONDS
  local tmp
  tmp="$(mktemp)"
  set +e
  # Ignore CMakeCache comment lines (which can mention "/opt/rocm" as the
  # upstream default in help text, even when the actual cache variables are
  # correctly pointing at in-tree prefixes).
  find "${ROOT}/${BUILD_DIR}" -name CMakeCache.txt -print0 2>/dev/null \
    | xargs -0 rg -nH "/opt/rocm" 2>/dev/null \
    | rg -v ":[0-9]+://" 2>/dev/null \
    >"${tmp}"
  local rc=$?
  set -e
  if [[ -n "${filter}" && -s "${tmp}" ]]; then
    # Remove known-benign matches (e.g. internal externalproject caches) for the light check.
    rg -v "${filter}" "${tmp}" > "${tmp}.filtered" 2>/dev/null || true
    mv -f "${tmp}.filtered" "${tmp}"
    if [[ ! -s "${tmp}" ]]; then
      rc=1
    else
      rc=0
    fi
  fi
  local elapsed=$((SECONDS - start))
  if [[ ${rc} -eq 0 ]]; then
    add_result "${label}" "FAIL" "${elapsed}s" "$(head -n 3 "${tmp}" | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
    rm -f "${tmp}"
    return 1
  fi
  add_result "${label}" "OK" "${elapsed}s" "${expected}"
  rm -f "${tmp}"
  return 0
}

check_toolchain_paths() {
  local stage_expect="$1"
  local label_prefix="$2"

  # Top-level cache compilers
  local cache="${ROOT}/${BUILD_DIR}/CMakeCache.txt"
  if [[ -f "${cache}" ]]; then
    local cxx
    cxx="$(rg -n "^CMAKE_CXX_COMPILER:[A-Z_]+=" "${cache}" | head -n 1 | cut -d= -f2- || true)"
    if [[ -n "${cxx}" ]]; then
      if [[ "${stage_expect}" == "stage2" ]]; then
        case "${cxx}" in
          *"${ROOT}/${STAGE1_BUILD_DIR}/"*)
            add_result "${label_prefix} top-level compiler" "OK" "0s" "${cxx}"
            ;;
          *)
            add_result "${label_prefix} top-level compiler" "FAIL" "0s" "expected Stage-1 toolchain clang++ from ${STAGE1_BUILD_DIR}, got: ${cxx}"
            ;;
        esac
      else
        add_result "${label_prefix} top-level compiler" "OK" "0s" "${cxx}"
      fi
    else
      add_result "${label_prefix} top-level compiler" "SKIP" "0s" "no CMAKE_CXX_COMPILER in cache"
    fi
  else
    add_result "${label_prefix} top-level compiler" "SKIP" "0s" "missing ${BUILD_DIR}/CMakeCache.txt"
  fi

  # Scan toolchain files for system clang fallbacks (Stage-2 should not contain these).
  if [[ "${stage_expect}" == "stage2" ]]; then
    local maxdepth_args=()
    if (( CONSISTENCY_DEEP == 0 )); then
      maxdepth_args=(-maxdepth 6)
    fi
    local tmp
    tmp="$(mktemp)"
    set +e
    find "${ROOT}/${BUILD_DIR}" "${maxdepth_args[@]}" -name "*_toolchain.cmake" -print0 2>/dev/null | \
      xargs -0 rg -nH "/usr/lib/llvm-18/|/usr/bin/clang\\+\\+|/usr/bin/clang(\\s|$)" 2>/dev/null >"${tmp}"
    local rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      add_result "${label_prefix} toolchain scan" "FAIL" "0s" "$(head -n 3 "${tmp}" | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
      rm -f "${tmp}"
      return 1
    fi
    add_result "${label_prefix} toolchain scan" "OK" "0s" "no system clang paths found"
    rm -f "${tmp}"
  else
    add_result "${label_prefix} toolchain scan" "SKIP" "0s" "stage1/system clang allowed"
  fi

  # compile_commands.json scan (if present)
  local cc_file=""
  if [[ -f "${ROOT}/${BUILD_DIR}/compile_commands.json" ]]; then
    cc_file="${ROOT}/${BUILD_DIR}/compile_commands.json"
  elif [[ -f "${ROOT}/compile_commands.json" ]]; then
    cc_file="${ROOT}/compile_commands.json"
  fi
  if [[ -n "${cc_file}" ]]; then
    if [[ "${stage_expect}" == "stage2" ]]; then
      check_no_matches_in_files "${label_prefix} compile_commands" "no system clang in ${cc_file}" "/usr/lib/llvm-18/|/usr/bin/clang\\+\\+" "${cc_file}" || true
    else
      add_result "${label_prefix} compile_commands" "SKIP" "0s" "stage1/system clang allowed (${cc_file})"
    fi
  else
    add_result "${label_prefix} compile_commands" "SKIP" "0s" "no compile_commands.json found"
  fi
}

check_hip_device_libs() {
  local label_prefix="$1"
  if [[ ! -d "${HIP_DEVICE_LIB_PATH}" ]]; then
    add_result "${label_prefix} hip device libs" "FAIL" "0s" "HIP_DEVICE_LIB_PATH not found: ${HIP_DEVICE_LIB_PATH}"
    return 1
  fi
  if [[ -f "${HIP_DEVICE_LIB_PATH}/oclc_isa_version_1031.bc" ]]; then
    add_result "${label_prefix} hip device libs" "OK" "0s" "found oclc_isa_version_1031.bc"
  else
    add_result "${label_prefix} hip device libs" "FAIL" "0s" "missing oclc_isa_version_1031.bc under ${HIP_DEVICE_LIB_PATH}"
  fi

  if command -v hipcc >/dev/null 2>&1 && [[ -f "${ROOT}/test_hip.cpp" ]]; then
    local tmp
    tmp="$(mktemp)"
    set +e
    hipcc -v --offload-arch=gfx1031 "${ROOT}/test_hip.cpp" -o "${tmp}.out" 2>"${tmp}"
    local rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
      add_result "${label_prefix} hipcc -v" "FAIL" "0s" "hipcc failed (rc=${rc})"
      rm -f "${tmp}" "${tmp}.out" 2>/dev/null || true
      return 1
    fi
    if rg -q "gfx1031|oclc_isa_version_1031|amdgcn/bitcode" "${tmp}"; then
      add_result "${label_prefix} hipcc -v" "OK" "0s" "saw gfx1031/device-lib path in hipcc -v"
    else
      add_result "${label_prefix} hipcc -v" "FAIL" "0s" "no gfx1031/device-lib hints in hipcc -v"
    fi
    rm -f "${tmp}" "${tmp}.out" 2>/dev/null || true
  else
    add_result "${label_prefix} hipcc -v" "SKIP" "0s" "hipcc or test_hip.cpp missing"
  fi
}

check_runtime_linkage() {
  local label_prefix="$1"
  local -a bins=(rocminfo hipinfo rocblas-bench hipblas-bench)
  local b
  for b in "${bins[@]}"; do
    if ! command -v "${b}" >/dev/null 2>&1; then
      add_result "${label_prefix} ldd ${b}" "SKIP" "0s" "not in PATH"
      continue
    fi
    local exe
    exe="$(command -v "${b}")"
    local tmp
    tmp="$(mktemp)"
    set +e
    ldd "${exe}" 2>/dev/null | rg -n "/opt/rocm" >"${tmp}"
    local rc=$?
    set -e
    if [[ ${rc} -eq 0 ]]; then
      add_result "${label_prefix} ldd ${b}" "FAIL" "0s" "$(head -n 2 "${tmp}" | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
      rm -f "${tmp}"
      continue
    fi
    add_result "${label_prefix} ldd ${b}" "OK" "0s" "no /opt/rocm deps"
    rm -f "${tmp}"
  done
}

find_rocm_lib() {
  local soname="$1"
  local p
  for p in \
    "${ROCM_PATH}/lib/${soname}" \
    "${ROCM_PATH}/lib64/${soname}" \
    "${ROCM_PATH}/lib/${soname}".* \
    "${ROCM_PATH}/lib64/${soname}".*; do
    if compgen -G "${p}" >/dev/null 2>&1; then
      compgen -G "${p}" | head -n 1
      return 0
    fi
  done
  return 1
}

check_rocm_core_components() {
  local label_prefix="$1"
  local lib

  if (( HAVE_ROCM_ENV == 0 )); then
    add_result "${label_prefix} HIP runtime" "SKIP" "0s" "ROCM_PATH missing"
    add_result "${label_prefix} HSA runtime" "SKIP" "0s" "ROCM_PATH missing"
    add_result "${label_prefix} rocBLAS" "SKIP" "0s" "ROCM_PATH missing"
    add_result "${label_prefix} MIOpen" "SKIP" "0s" "ROCM_PATH missing"
    return 0
  fi

  if lib="$(find_rocm_lib "libamdhip64.so")"; then
    add_result "${label_prefix} HIP runtime" "OK" "0s" "${lib}"
  else
    add_result "${label_prefix} HIP runtime" "FAIL" "0s" "libamdhip64.so not found under ${ROCM_PATH}/lib{,64}"
  fi

  if lib="$(find_rocm_lib "libhsa-runtime64.so")"; then
    add_result "${label_prefix} HSA runtime" "OK" "0s" "${lib}"
  else
    add_result "${label_prefix} HSA runtime" "FAIL" "0s" "libhsa-runtime64.so not found under ${ROCM_PATH}/lib{,64}"
  fi

  if lib="$(find_rocm_lib "librocblas.so")"; then
    add_result "${label_prefix} rocBLAS" "OK" "0s" "${lib}"
  else
    add_result "${label_prefix} rocBLAS" "FAIL" "0s" "librocblas.so not found under ${ROCM_PATH}/lib{,64}"
  fi

  if lib="$(find_rocm_lib "libMIOpen.so")"; then
    add_result "${label_prefix} MIOpen" "OK" "0s" "${lib}"
  else
    add_result "${label_prefix} MIOpen" "SKIP" "0s" "libMIOpen.so not found (component may be disabled)"
  fi

  if command -v rocblas-bench >/dev/null 2>&1; then
    run_timed "${label_prefix} rocBLAS --version" "typ. <1s" rocblas-bench --version || true
  else
    add_result "${label_prefix} rocBLAS --version" "SKIP" "0s" "rocblas-bench not in PATH"
  fi

  local drv=""
  if drv="$(miopen_find_driver)"; then
    run_timed "${label_prefix} MIOpen --version" "typ. <1s" "${drv}" --version || true
  else
    add_result "${label_prefix} MIOpen --version" "SKIP" "0s" "MIOpenDriver/miopen-driver not in PATH"
  fi
}

run_rocm_core_component_load_tests() {
  local label_prefix="$1"
  shift
  local -a selected=("$@")
  local timeout_s="${BENCH_TIMEOUT_S:-300}"
  local expected_load="typ. 5-30s"
  local hip_n hip_reps hip_inner
  local rb_m rb_n rb_k rb_iters rb_cold
  local mi_n mi_c mi_h mi_w mi_k mi_y mi_x mi_reps

  if [[ "${MODE}" == "full" ]]; then
    hip_n="${CORE_HIP_N:-16777216}"
    hip_reps="${CORE_HIP_REPS:-900}"
    hip_inner="${CORE_HIP_INNER:-160}"
    rb_m="${CORE_ROCBLAS_M:-6144}"
    rb_n="${CORE_ROCBLAS_N:-6144}"
    rb_k="${CORE_ROCBLAS_K:-6144}"
    rb_iters="${CORE_ROCBLAS_ITERS:-120}"
    rb_cold="${CORE_ROCBLAS_COLD_ITERS:-8}"
    mi_n="${CORE_MIOPEN_N:-32}"
    mi_c="${CORE_MIOPEN_C:-64}"
    mi_h="${CORE_MIOPEN_H:-128}"
    mi_w="${CORE_MIOPEN_W:-128}"
    mi_k="${CORE_MIOPEN_K:-128}"
    mi_y="${CORE_MIOPEN_Y:-3}"
    mi_x="${CORE_MIOPEN_X:-3}"
    mi_reps="${CORE_MIOPEN_REPS:-12}"
  else
    hip_n="${CORE_HIP_N:-8388608}"
    hip_reps="${CORE_HIP_REPS:-2000}"
    hip_inner="${CORE_HIP_INNER:-128}"
    rb_m="${CORE_ROCBLAS_M:-4096}"
    rb_n="${CORE_ROCBLAS_N:-4096}"
    rb_k="${CORE_ROCBLAS_K:-4096}"
    rb_iters="${CORE_ROCBLAS_ITERS:-420}"
    rb_cold="${CORE_ROCBLAS_COLD_ITERS:-6}"
    mi_n="${CORE_MIOPEN_N:-16}"
    mi_c="${CORE_MIOPEN_C:-64}"
    mi_h="${CORE_MIOPEN_H:-112}"
    mi_w="${CORE_MIOPEN_W:-112}"
    mi_k="${CORE_MIOPEN_K:-128}"
    mi_y="${CORE_MIOPEN_Y:-3}"
    mi_x="${CORE_MIOPEN_X:-3}"
    mi_reps="${CORE_MIOPEN_REPS:-16}"
  fi

  load_selected() {
    local idx="$1"
    if (( ${#selected[@]} == 0 )); then
      return 0
    fi
    local s
    for s in "${selected[@]}"; do
      if [[ "${s}" == "${idx}" ]]; then
        return 0
      fi
    done
    return 1
  }

  if (( HAVE_ROCM_ENV == 0 )); then
    if load_selected 10; then
      add_result "${label_prefix} HIP/HSA load" "SKIP" "0s" "ROCM_PATH missing"
    fi
    if load_selected 11; then
      add_result "${label_prefix} rocBLAS load" "SKIP" "0s" "ROCM_PATH missing"
    fi
    if load_selected 12; then
      add_result "${label_prefix} MIOpen load" "SKIP" "0s" "ROCM_PATH missing"
    fi
    return 0
  fi

  if load_selected 10 && command -v hipcc >/dev/null 2>&1; then
    BENCH_META_STATS=""
    local hip_ops hip_bytes
    hip_ops="$(awk -v n="${hip_n}" -v r="${hip_reps}" -v inr="${hip_inner}" 'BEGIN{printf "%.0f", 2.0*n*inr*r}')"
    hip_bytes="$(awk -v n="${hip_n}" -v r="${hip_reps}" 'BEGIN{printf "%.0f", 16.0*n*r}')"
    BENCH_META_STATS="$(set_bench_meta_stats "${hip_ops}" "FLOP" "${hip_bytes}")"
    run_bench_with_timeout "bench: ${label_prefix} HIP/HSA load" "${expected_load}" "${timeout_s}" \
      env CORE_HIP_N="${hip_n}" CORE_HIP_REPS="${hip_reps}" CORE_HIP_INNER="${hip_inner}" \
      bash -lc '
        set -euo pipefail
        src="$(mktemp /tmp/hip_hsa_load.XXXXXX.cpp)"
        bin="${src%.cpp}.out"
        trap "rm -f \"$src\" \"$bin\"" EXIT
        cat >"$src" <<'"'"'CPP'"'"'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void fma_stress(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n,
                           int inner) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  float y = b[i];
  float z = c[i];
  for (int t = 0; t < inner; ++t) {
    z = fmaf(x, y, z);
    x = x * 1.000001f + 0.000001f;
    y = y * 0.999999f + 0.000002f;
  }
  c[i] = z;
}

int main(int argc, char** argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 23);
  int reps = (argc > 2) ? std::atoi(argv[2]) : 220;
  int inner = (argc > 3) ? std::atoi(argv[3]) : 128;
  if (n <= 0 || reps <= 0 || inner <= 0) return 2;

  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  if (hipMalloc(&da, bytes) != hipSuccess) return 3;
  if (hipMalloc(&db, bytes) != hipSuccess) return 4;
  if (hipMalloc(&dc, bytes) != hipSuccess) return 5;
  if (hipMemset(da, 0x3f, bytes) != hipSuccess) return 6;
  if (hipMemset(db, 0x40, bytes) != hipSuccess) return 7;
  if (hipMemset(dc, 0x00, bytes) != hipSuccess) return 8;

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  for (int r = 0; r < reps; ++r) {
    hipLaunchKernelGGL(fma_stress, grid, block, 0, 0, da, db, dc, n, inner);
  }
  if (hipDeviceSynchronize() != hipSuccess) return 9;
  float out = 0.0f;
  if (hipMemcpy(&out, dc, sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 10;
  if (!(out == out)) return 11;
  hipFree(da);
  hipFree(db);
  hipFree(dc);
  return 0;
}
CPP
        hipcc --offload-arch=gfx1031 -O3 "$src" -o "$bin"
        "$bin" "${CORE_HIP_N}" "${CORE_HIP_REPS}" "${CORE_HIP_INNER}" >/dev/null
      ' || true
    BENCH_META_STATS=""
  elif load_selected 10; then
    add_result "${label_prefix} HIP/HSA load" "SKIP" "0s" "hipcc not in PATH"
  fi

  if load_selected 11 && command -v rocblas-bench >/dev/null 2>&1; then
    BENCH_META_STATS=""
    local rb_ops rb_bytes
    rb_ops="$(awk -v m="${rb_m}" -v n="${rb_n}" -v k="${rb_k}" -v it="${rb_iters}" 'BEGIN{printf "%.0f", 2.0*m*n*k*it}')"
    rb_bytes="$(awk -v m="${rb_m}" -v n="${rb_n}" -v k="${rb_k}" -v it="${rb_iters}" 'BEGIN{printf "%.0f", (m*k + k*n + 2.0*m*n)*4.0*it}')"
    BENCH_META_STATS="$(set_bench_meta_stats "${rb_ops}" "FLOP" "${rb_bytes}")"
    run_bench_with_timeout "bench: ${label_prefix} rocBLAS load" "${expected_load}" "${timeout_s}" \
      rocblas-bench -f gemm -r f32_r -m "${rb_m}" -n "${rb_n}" -k "${rb_k}" --alpha 1.0 --beta 0.0 --iters "${rb_iters}" --cold_iters "${rb_cold}" || true
    BENCH_META_STATS=""
  elif load_selected 11; then
    add_result "${label_prefix} rocBLAS load" "SKIP" "0s" "rocblas-bench not in PATH"
  fi

  local drv=""
  if load_selected 12 && drv="$(miopen_find_driver)"; then
    BENCH_META_STATS=""
    local mi_ops mi_bytes
    mi_ops="$(awk -v n="${mi_n}" -v h="${mi_h}" -v w="${mi_w}" -v k="${mi_k}" -v c="${mi_c}" -v y="${mi_y}" -v x="${mi_x}" -v r="${mi_reps}" 'BEGIN{printf "%.0f", 2.0*n*h*w*k*c*y*x*r}')"
    mi_bytes="$(awk -v n="${mi_n}" -v h="${mi_h}" -v w="${mi_w}" -v k="${mi_k}" -v c="${mi_c}" -v y="${mi_y}" -v x="${mi_x}" -v r="${mi_reps}" 'BEGIN{inb=n*c*h*w; wtb=k*c*y*x; outb=n*k*h*w; printf "%.0f", (inb+wtb+outb)*4.0*r}')"
    BENCH_META_STATS="$(set_bench_meta_stats "${mi_ops}" "FLOP" "${mi_bytes}")"
    run_bench_with_timeout "bench: ${label_prefix} MIOpen load" "${expected_load}" "${timeout_s}" \
      env CORE_MIOPEN_REPS="${mi_reps}" \
      bash -lc '
        set -euo pipefail
        drv="$1"; shift
        for _ in $(seq 1 "${CORE_MIOPEN_REPS}"); do
          "$drv" conv "$@" >/dev/null
        done
      ' _ "${drv}" -n "${mi_n}" -c "${mi_c}" -H "${mi_h}" -W "${mi_w}" -k "${mi_k}" -y "${mi_y}" -x "${mi_x}" -p 1 -q 1 || true
    BENCH_META_STATS=""
  elif load_selected 12; then
    add_result "${label_prefix} MIOpen load" "SKIP" "0s" "MIOpenDriver/miopen-driver not in PATH"
  fi
}

extract_gflops() {
  local file="$1"
  local gflops
  # rocblas-bench / hipblas-bench CSV format:
  # header contains rocblas-Gflops or hipblas-Gflops and data row is comma-separated.
  gflops=$(awk -F',' '
    BEGIN { col=0 }
    /(^|,)rocblas-Gflops(,|$)/ || /(^|,)hipblas-Gflops(,|$)/ {
      for(i=1;i<=NF;i++) {
        if($i ~ /rocblas-Gflops/ || $i ~ /hipblas-Gflops/) { col=i; break }
      }
      next
    }
    col>0 && ($0 ~ /^[[:space:]]*[NTC],[NTC],/ || $0 ~ /^[[:space:]]*[a-zA-Z0-9_]+,[NTC],[NTC],/) {
      v=$col
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      if(v ~ /^[0-9]+(\.[0-9]+)?$/) val=v
    }
    END { if(val!="") print val }
  ' "$file" 2>/dev/null)
  if [[ -n "${gflops}" ]]; then
    echo "${gflops}"
    return 0
  fi
  gflops=$(grep -Eo '([0-9]+(\.[0-9]+)?)\s*(Gflop/s|GFLOP/s|GFLOPS|gflops)' "$file" | tail -n1 | awk '{print $1}')
  if [[ -n "${gflops}" ]]; then
    echo "${gflops}"
    return 0
  fi
  gflops=$(awk '
    /Gflop/ {for(i=1;i<=NF;i++) if($i ~ /Gflop/) col=i}
    col && NF>=col {val=$col}
    END {if(val!="") print val}
  ' "$file")
  if [[ -n "${gflops}" ]]; then
    echo "${gflops}"
    return 0
  fi
  echo ""
}

extract_tflops() {
  local file="$1"
  local tflops
  tflops=$(grep -Eo '([0-9]+(\.[0-9]+)?)\s*(Tflop/s|TFLOP/s|TFLOPS|tflops)' "$file" | tail -n1 | awk '{print $1}')
  if [[ -n "${tflops}" ]]; then
    echo "${tflops}"
    return 0
  fi
  echo ""
}

extract_gbps() {
  local file="$1"
  local v
  v=$(grep -Eo '([0-9]+(\.[0-9]+)?)\s*GB/s' "$file" | tail -n1 | awk '{print $1}')
  echo "${v}"
}

extract_gsamples() {
  local file="$1"
  local v
  v=$(grep -Eo '([0-9]+(\.[0-9]+)?)\s*GSample/s' "$file" | tail -n1 | awk '{print $1}')
  echo "${v}"
}

extract_ms() {
  local file="$1"
  local v
  v=$(grep -Eo '([0-9]+(\.[0-9]+)?)\s*ms' "$file" | tail -n1 | awk '{print $1}')
  echo "${v}"
}

extract_sparse_metrics() {
  local file="$1"
  awk '
    BEGIN { g=0; b=0; m=0; gf=""; gb=""; ms="" }
    /^size[[:space:]]/ {
      for(i=1;i<=NF;i++) {
        if($i=="GFlop/s" || $i=="GFlops") g=i
        if($i=="GB/s") b=i
        if($i=="msec" || $i=="ms") m=i
      }
      next
    }
    g>0 && $1 ~ /^[0-9]+/ {
      gf=$g
      if(b>0) gb=$b
      if(m>0) ms=$m
    }
    END {
      if(gf!="") {
        out=""
        if(gb!="") out="GB/s=" gb
        if(out!="" && gf!="") out=out " "
        out=out "GFLOP/s=" gf
        print out
      }
    }
  ' "$file" 2>/dev/null
}

extract_rocrand_metrics() {
  local file="$1"
  awk -F',' '
    BEGIN { gb=""; gs=""; ms="" }
    /^[a-zA-Z0-9_]+,[a-zA-Z0-9_-]+,[0-9]/ {
      gb=$3; gs=$4; ms=$5
    }
    END {
      if(gb!="") {
        out="GB/s=" gb
        if(gs!="") out=out " GSample/s=" gs
        print out
      }
    }
  ' "$file" 2>/dev/null
}

extract_single_number() {
  local file="$1"
  local v
  # Some bench clients print a single numeric value (e.g. gpu_time_us).
  v="$(tr -d '[:space:]' <"${file}" | head -c 64)"
  if [[ "${v}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "${v}"
    return 0
  fi
  echo ""
  return 1
}

print_bench_menu() {
  cat <<'EOF_BENCH_MENU'

Bench menu (gfx1031):
  0) Run all (1-12)
  1) rocBLAS GEMM f32
  2) hipBLAS GEMM f32
  3) rocSOLVER geqrf_strided_batched (s)
  4) hipSOLVER (default) tiny solver
  5) rocSPARSE axpyi (s)
  6) hipSPARSE axpyi (s)
  7) rocFFT complex fwd (sustained)
  8) dyna-rocFFT complex fwd (sustained)
  9) rocRAND generate (philox, uniform-float)
 10) Component load: HIP/HSA
 11) Component load: rocBLAS
 12) Component load: MIOpen

Enter one number (e.g. 2) or a list (e.g. 1,2,7). Use 'q' to quit.
Default ROCm source: in-tree (<repo>/<build-dir>/dist/rocm).
Use --system-rocm or --system-rocm-path <dir> only when you explicitly want system ROCm.
EOF_BENCH_MENU
}

parse_bench_selection() {
  local input="$1"
  local -a out=()
  input="$(echo "${input}" | tr -d '[:space:]')"
  input="${input//,/ }"
  # shellcheck disable=SC2206
  out=(${input})
  echo "${out[@]}"
}

fmt_status() {
  local s="$1"
  case "${s}" in
    OK) echo "${C_GREEN}OK${C_RESET}" ;;
    FAIL) echo "${C_RED}FAIL${C_RESET}" ;;
    SKIP) echo "${C_YELLOW}SKIP${C_RESET}" ;;
    *) echo "${s}" ;;
  esac
}

fmt_label() {
  local s="$1"
  echo "${C_BOLD}${s}${C_RESET}"
}

now_ms() {
  local t
  t="$(date +%s%3N 2>/dev/null || true)"
  if [[ -n "${t}" && "${t}" =~ ^[0-9]+$ ]]; then
    echo "${t}"
    return 0
  fi
  echo "$(( $(date +%s) * 1000 ))"
}

fmt_duration_ms() {
  local ms="$1"
  if [[ -z "${ms}" || "${ms}" == "0" ]]; then
    echo "0ms"
    return 0
  fi
  if (( ms < 1000 )); then
    echo "${ms}ms"
    return 0
  fi
  local s=$((ms / 1000))
  local rem_ms=$((ms % 1000))
  if (( s < 60 )); then
    printf "%d.%03ds" "${s}" "${rem_ms}"
    return 0
  fi
  local m=$((s / 60))
  local rem_s=$((s % 60))
  printf "%dm%02ds" "${m}" "${rem_s}"
}

print_run_header() {
  local title="$1"
  local build_dir="$2"
  local rocm_path="$3"
  local rocm_source="in-tree"
  if (( USE_SYSTEM_ROCM )); then
    rocm_source="system"
  fi

  local log_state="disabled (use --log [file])"
  if (( LOG_ENABLED )); then
    log_state="${LOG_FILE}"
  fi

  echo "${C_BOLD}${title}${C_RESET}" | tee -a "${LOG_FILE}"
  echo "${C_DIM}- build dir:${C_RESET} ${build_dir}" | tee -a "${LOG_FILE}"
  echo "${C_DIM}- ROCm:${C_RESET} ${rocm_path}" | tee -a "${LOG_FILE}"
  echo "${C_DIM}- ROCm source:${C_RESET} ${rocm_source}" | tee -a "${LOG_FILE}"
  echo "${C_DIM}- mode:${C_RESET} ${MODE} (BENCH_SIZE=${BENCH_SIZE:-auto}, BENCH_ITERS=${BENCH_ITERS:-auto})" | tee -a "${LOG_FILE}"
  echo "${C_DIM}- logging:${C_RESET} ${log_state}" | tee -a "${LOG_FILE}"
  if (( RUN_POWER )); then
    echo "${C_DIM}- power:${C_RESET} enabled (sysfs)" | tee -a "${LOG_FILE}"
  fi

  local what=()
  if (( RUN_CONSISTENCY )); then what+=("consistency"); fi
  if (( RUN_MIOPEN )); then
    if (( RUN_MIOPEN_SMOKE )); then what+=("miopen-smoke"); else what+=("miopen"); fi
  fi
  if (( RUN_BENCH )); then
    if (( BENCH_MENU )); then what+=("bench-menu"); elif (( BENCH_LITE )); then what+=("bench-lite"); else what+=("bench"); fi
  fi
  if (( ${#what[@]} )); then
    echo "${C_DIM}- will run:${C_RESET} ${what[*]}" | tee -a "${LOG_FILE}"
  fi
  echo "" | tee -a "${LOG_FILE}"
}

POWER_PATH=""
GPU_BUSY_PATH=""
MEM_BUSY_PATH=""
POWER_SAMPLER_PID=""

discover_power_sensor() {
  POWER_PATH=""
  GPU_BUSY_PATH=""
  MEM_BUSY_PATH=""
  local drm="/sys/class/drm"
  [[ -d "${drm}" ]] || return 1
  local card
  for card in "${drm}"/card[0-9]*; do
    [[ -d "${card}" ]] || continue
    local dev="${card}/device"
    [[ -d "${dev}/hwmon" ]] || continue
    local hm
    for hm in "${dev}/hwmon"/hwmon*; do
      [[ -d "${hm}" ]] || continue
      local name=""
      if [[ -f "${hm}/name" ]]; then
        name="$(cat "${hm}/name" 2>/dev/null || true)"
        name="${name//$'\n'/}"
      fi
      [[ "${name}" == "amdgpu" ]] || continue
      if [[ -f "${hm}/power1_average" ]]; then
        POWER_PATH="${hm}/power1_average"
        [[ -f "${dev}/gpu_busy_percent" ]] && GPU_BUSY_PATH="${dev}/gpu_busy_percent"
        [[ -f "${dev}/mem_busy_percent" ]] && MEM_BUSY_PATH="${dev}/mem_busy_percent"
        return 0
      fi
    done
  done
  return 1
}

power_sampler_start() {
  local out_file="$1"
  local interval_s="${2:-0.5}"
  : >"${out_file}"
  (
    set +e
    read_quick() {
      local p="$1"
      cat "${p}" 2>/dev/null || true
    }
    while true; do
      local t_ms
      t_ms="$(now_ms)"
      local p_uw=""
      local gpu=""
      local mem=""
      if [[ -n "${POWER_PATH}" ]] && [[ -r "${POWER_PATH}" ]]; then
        p_uw="$(read_quick "${POWER_PATH}")"
      fi
      if [[ -n "${GPU_BUSY_PATH}" ]] && [[ -r "${GPU_BUSY_PATH}" ]]; then
        gpu="$(read_quick "${GPU_BUSY_PATH}")"
      fi
      if [[ -n "${MEM_BUSY_PATH}" ]] && [[ -r "${MEM_BUSY_PATH}" ]]; then
        mem="$(read_quick "${MEM_BUSY_PATH}")"
      fi
      echo "${t_ms} ${p_uw:-} ${gpu:-} ${mem:-}" >>"${out_file}"
      sleep "${interval_s}"
    done
  ) &
  POWER_SAMPLER_PID="$!"
}

power_sampler_stop_and_format() {
  local out_file="$1"
  local pid="$2"
  local baseline_avg_w="$3" # may be empty
  if [[ -n "${pid}" ]]; then
    kill "${pid}" >/dev/null 2>&1 || true
    # Avoid blocking forever if the sampler is stuck in an uninterruptible read.
    for _ in {1..10}; do
      kill -0 "${pid}" >/dev/null 2>&1 || break
      sleep 0.05
    done
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  fi
  awk -v base="${baseline_avg_w:-}" '
  function fmt_wh(v){ if(v=="") return "   n/a"; return sprintf("%.3fWh", v/3600.0) }
  function fmt_w(v){ if(v=="") return "   n/a"; return sprintf("%6.1fW", v) }
  function fmt_dw(v){ if(v=="") return "   n/a"; return sprintf("%+6.1fW", v) }
  function fmt_pct(v){ if(v=="") return "n/a"; return sprintf("%3.0f", v) }
  BEGIN{prev_t=""; prev_p=""; n=0; sum_p=0; max_p=""; sum_gpu=0; n_gpu=0; sum_mem=0; n_mem=0; e=0}
  /^[0-9]+/{
    t=$1
    p_uw=$2
    gpu=$3
    mem=$4
    p=""
    if(p_uw ~ /^[0-9]+$/){ p=p_uw/1000000.0; sum_p+=p; n++; if(max_p==""||p>max_p) max_p=p }
    if(gpu ~ /^[0-9]+$/){ sum_gpu+=gpu; n_gpu++ }
    if(mem ~ /^[0-9]+$/){ sum_mem+=mem; n_mem++ }
    if(prev_t!="" && prev_p!="" && p!=""){
      dt=(t-prev_t)/1000.0
      e += 0.5*(prev_p+p)*dt
    }
    prev_t=t
    if(p!=""){ prev_p=p }
  }
  END{
    avg=""; dw=""
    if(n>0){ avg=sum_p/n }
    if(avg!="" && base!="" && base ~ /^[0-9.]+$/){ dw=avg-base }
    gpu_avg=""; mem_avg=""
    if(n_gpu>0){ gpu_avg=sum_gpu/n_gpu }
    if(n_mem>0){ mem_avg=sum_mem/n_mem }
    if(n<2){ e="" }
    printf("E=%s  avgW=%s  dW=%s  maxW=%s  gpu%%=%s  mem%%=%s", fmt_wh(e), fmt_w(avg), fmt_dw(dw), fmt_w(max_p), fmt_pct(gpu_avg), fmt_pct(mem_avg))
  }' "${out_file}"
}

power_compute_avg_w() {
  local out_file="$1"
  awk '
  BEGIN{n=0; sum=0}
  /^[0-9]+/{
    p_uw=$2
    if(p_uw ~ /^[0-9]+$/){ sum += (p_uw/1000000.0); n++ }
  }
  END{ if(n>0) printf("%.3f", sum/n) }' "${out_file}"
}

power_wrap() {
  local label="$1"
  local expected="$2"
  local timeout_s="$3" # may be empty
  shift 3
  local -a cmd=("$@")

  local tmp
  tmp="$(mktemp)"
  local ptmp
  ptmp="$(mktemp)"

  # For each test: capture a 5s idle baseline (no GPU load) so dW is meaningful
  # even when runs are short or bursty.
  local baseline_avg_w=""
  if (( RUN_POWER )); then
    if [[ -z "${POWER_PATH}" ]]; then
      discover_power_sensor || true
    fi
    if [[ -n "${POWER_PATH}" ]]; then
      local btmp
      btmp="$(mktemp)"
      power_sampler_start "${btmp}" "0.2"
      local bpid="${POWER_SAMPLER_PID}"
      sleep 5
      baseline_avg_w="$(power_compute_avg_w "${btmp}")"
      _="$(power_sampler_stop_and_format "${btmp}" "${bpid}" "")" || true
      rm -f "${btmp}"
    fi
  fi

  local start_ms
  start_ms="$(now_ms)"
  power_sampler_start "${ptmp}" "0.2"
  local pid="${POWER_SAMPLER_PID}"

  set +e
  if [[ -n "${timeout_s}" ]] && command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${timeout_s}" "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}" | tee "${tmp}" >/dev/null
  else
    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}" | tee "${tmp}" >/dev/null
  fi
  local rc=${PIPESTATUS[0]}
  set -e
  local end_ms
  end_ms="$(now_ms)"
  local elapsed_ms=$((end_ms - start_ms))

  local power_blob=""
  if [[ -n "${POWER_PATH}" ]]; then
    power_blob="$(power_sampler_stop_and_format "${ptmp}" "${pid}" "${baseline_avg_w}")"
  else
    [[ -n "${pid}" ]] && kill "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${ptmp}"

  local metric=""
  if [[ ${rc} -eq 124 || ${rc} -eq 137 || ${rc} -eq 143 ]]; then
    metric="timeout after ${timeout_s}s (first run may JIT; rerun)"
    [[ -n "${power_blob}" ]] && metric="${metric} | ${power_blob}"
    add_result "${label}" "SKIP" "$(fmt_duration_ms "${elapsed_ms}")" "${metric}"
    rm -f "${tmp}"
    return 0
  elif [[ ${rc} -eq 0 ]]; then
    local gflops
    local tflops
    gflops="$(extract_gflops "${tmp}")"
    tflops="$(extract_tflops "${tmp}")"
    if [[ -z "${tflops}" && -n "${gflops}" ]]; then
      tflops=$(awk -v v="${gflops}" 'BEGIN{printf "%.3f", v/1000.0}')
      metric="TFLOPS=${tflops}"
    elif [[ -n "${tflops}" ]]; then
      metric="TFLOPS=${tflops}"
    fi
    if [[ -z "${metric}" ]]; then
      local gbps
      local gs
      local ms
      local special
      gbps="$(extract_gbps "${tmp}")"
      gs="$(extract_gsamples "${tmp}")"
      ms="$(extract_ms "${tmp}")"
      if [[ -n "${gbps}" ]]; then
        metric="GB/s=${gbps}"
        if [[ -n "${gs}" ]]; then
          metric="${metric} GSample/s=${gs}"
        fi
      elif [[ -n "${gs}" ]]; then
        metric="GSample/s=${gs}"
      elif [[ -n "${ms}" ]]; then
        metric="ms=${ms}"
      fi
      if [[ -z "${metric}" && ( "${label}" == *"rocSPARSE"* || "${label}" == *"hipSPARSE"* ) ]]; then
        special="$(extract_sparse_metrics "${tmp}")"
        metric="${special:-}"
      fi
      if [[ -z "${metric}" && "${label}" == *"rocRAND"* ]]; then
        special="$(extract_rocrand_metrics "${tmp}")"
        metric="${special:-}"
      fi
      if [[ -z "${metric}" && ( "${label}" == *"rocSOLVER"* || "${label}" == *"hipSOLVER"* ) ]]; then
        special="$(extract_single_number "${tmp}")"
        if [[ -n "${special}" ]]; then
          metric="gpu_time_us=${special}"
        fi
      fi
    fi
    # If power sampling is disabled, avoid cluttering the output with both TFLOPS
    # and GFLOPS. Prefer TFLOPS.
    # Strip GFLOPS suffix if present (we print TFLOPS as the primary metric).
    if [[ "${metric}" == *"(GFLOPS="* ]]; then
      metric="${metric%% (GFLOPS=*}"
    fi
    [[ -n "${power_blob}" ]] && metric="${metric} | ${power_blob}"
    add_result "${label}" "OK" "$(fmt_duration_ms "${elapsed_ms}")" "${metric}"
  else
    metric="rc=${rc}"
    [[ -n "${power_blob}" ]] && metric="${metric} | ${power_blob}"
    add_result "${label}" "FAIL" "$(fmt_duration_ms "${elapsed_ms}")" "${metric}"
  fi
  rm -f "${tmp}"
  return ${rc}
}

run_timed() {
  local label="$1"
  local expected="$2"
  shift 2
  local cmd=("$@")
  local tmp
  tmp="$(mktemp)"
  echo "${C_CYAN}==>${C_RESET} $(fmt_label "${label}") ${C_DIM}(expected: ${expected})${C_RESET}" | tee -a "${LOG_FILE}"
  if (( RUN_POWER )) && [[ -n "${POWER_PATH}" ]]; then
    rm -f "${tmp}"
    power_wrap "${label}" "${expected}" "" "${cmd[@]}"
    return $?
  fi
  local start_ms
  start_ms="$(now_ms)"
  set +e
  "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}" | tee "${tmp}" >/dev/null
  local rc=${PIPESTATUS[0]}
  set -e
  local end_ms
  end_ms="$(now_ms)"
  local elapsed_ms=$((end_ms - start_ms))
  if [[ ${rc} -eq 0 ]]; then
    add_result "${label}" "OK" "$(fmt_duration_ms "${elapsed_ms}")" ""
  else
    add_result "${label}" "FAIL" "$(fmt_duration_ms "${elapsed_ms}")" "rc=${rc}"
  fi
  rm -f "${tmp}"
  return ${rc}
}

run_bench_with_timeout() {
  local label="$1"
  local expected="$2"
  local timeout_s="$3"
  shift 3
  local cmd=("$@")
  local tmp
  tmp="$(mktemp)"
  print_bench_header "${label}" "${expected}"
  # Math anchor + formula + short plain-language description (quiet) to make
  # it obvious what is computed without reading surrounding docs.
  if [[ "${label}" == bench:* ]]; then
    local anchor=""
    local formula=""
    local desc=""
    case "${label}" in
      "bench: rocBLAS GEMM f32"|"bench: hipBLAS GEMM f32")
        anchor="GEMM"
        formula="C ← α·A·B + β·C   (A∈ℝ^{m×k}, B∈ℝ^{k×n}, C∈ℝ^{m×n})"
        if [[ "${label}" == "bench: rocBLAS GEMM f32" ]]; then
          desc="Dense BLAS-3 matrix multiply-accumulate; high arithmetic intensity. Stresses FMA throughput and the memory hierarchy under sustained load."
        else
          desc="Same GEMM computation, but invoked through the hipBLAS API layer."
        fi
        ;;
      bench:\ rocSOLVER\ geqrf_strided_batched*)
        anchor="QR"
        formula="A = Q·R,   Qᵀ·Q = I"
        desc="Batched Householder QR factorization producing orthonormal Q and upper-triangular R; common in least-squares and orthogonalization."
        ;;
      bench:\ hipSOLVER*)
        anchor="LU"
        formula="P·A = L·U"
        desc="Dense LU factorization with partial pivoting (GETRF); fundamental for solving A·x=b and related decompositions."
        ;;
      bench:\ rocSPARSE\ axpyi*|bench:\ hipSPARSE\ axpyi*)
        anchor="AXP"
        formula="∀j∈[0,nnz):  y[iⱼ] ← y[iⱼ] + α·xⱼ"
        if [[ "${label}" == bench:\ rocSPARSE* ]]; then
          desc="Sparse indexed AXPY (scatter-add into y). Stresses irregular gather/scatter and bandwidth/latency under sustained updates."
        else
          desc="Same AXPYI computation, but invoked through the hipSPARSE API layer."
        fi
        ;;
      bench:\ rocFFT\ complex\ fwd*|bench:\ dyna-rocFFT\ complex\ fwd*)
        anchor="FFT"
        formula="Xₖ = ∑ₙ₌₀^{N−1} xₙ · e^{−2π i k n / N}"
        if [[ "${label}" == bench:\ rocFFT* ]]; then
          desc="Batched complex-to-complex forward FFT. Stresses radix kernels, twiddle-factor math, and global memory traffic typical for spectral workloads."
        else
          desc="Same forward FFT, but loaded via the dynamic-loader client (runtime library selection)."
        fi
        ;;
      bench:\ rocRAND\ generate*)
        anchor="RNG"
        formula="xᵢ ∼ U(0,1)"
        desc="GPU pseudorandom variate generation (Philox, counter-based); measures RNG state generation and output write throughput."
        ;;
      bench:\ component\ HIP/HSA\ load)
        anchor="HIP"
        formula="∀i: cᵢ ← FMA_loop(aᵢ, bᵢ, cᵢ; inner), repeated reps times"
        desc="Native HIP-runtime kernel stress test (direct hipLaunchKernelGGL path). Exercises HSA queueing, dispatch, synchronization, and sustained ALU pressure without BLAS wrappers."
        ;;
      bench:\ component\ rocBLAS\ load)
        anchor="GEMM"
        formula="C ← α·A·B + β·C   (A∈ℝ^{m×k}, B∈ℝ^{k×n}, C∈ℝ^{m×n})"
        desc="Sustained rocBLAS GEMM load with larger matrix/iteration settings than smoke-level checks to provide meaningful power/utilization and throughput signal."
        ;;
      bench:\ component\ MIOpen\ load)
        anchor="CONV"
        formula="Y[n,k,h,w] = ∑_{c,y,x} W[k,c,y,x] · X[n,c,h+y,w+x]"
        desc="Repeated MIOpen convolution-forward workload to stress tensor-kernel dispatch, runtime graph setup, and memory traffic on a realistic DL primitive."
        ;;
    esac
    if [[ -n "${anchor}" ]]; then
      print_bench_anchor "${anchor}"
      print_formula_line "${formula}"
      case "${anchor}" in
        GEMM) print_ops_data_model "GEMM" ;;
        QR) print_ops_data_model "QR" ;;
        LU) print_ops_data_model "LU" ;;
        AXP) print_ops_data_model "AXPYI" ;;
        FFT) print_ops_data_model "FFT" ;;
        RNG) print_ops_data_model "RNG" ;;
        HIP) print_ops_data_model "HIPLOAD" ;;
        CONV) print_ops_data_model "CONV" ;;
      esac
      print_bench_desc "${desc}"
      if [[ -n "${BENCH_META_STATS:-}" ]]; then
        print_colorized_text "${BENCH_META_STATS}"
      fi
    fi
  fi
  if (( RUN_POWER )) && [[ -n "${POWER_PATH}" ]]; then
    rm -f "${tmp}"
    power_wrap "${label}" "${expected}" "${timeout_s}" "${cmd[@]}"
    return $?
  fi
  local start_ms
  start_ms="$(now_ms)"
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${timeout_s}" "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}" | tee "${tmp}" >/dev/null
  else
    "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}" | tee "${tmp}" >/dev/null
  fi
  local rc=${PIPESTATUS[0]}
  set -e
  local end_ms
  end_ms="$(now_ms)"
  local elapsed_ms=$((end_ms - start_ms))
  local metric=""
  if [[ ${rc} -eq 124 || ${rc} -eq 137 || ${rc} -eq 143 ]]; then
    add_result "${label}" "SKIP" "$(fmt_duration_ms "${elapsed_ms}")" "timeout after ${timeout_s}s (first run may JIT; rerun)"
    rm -f "${tmp}"
    return 0
  elif [[ ${rc} -eq 0 ]]; then
    local gflops
    local tflops
    gflops="$(extract_gflops "${tmp}")"
    tflops="$(extract_tflops "${tmp}")"
    if [[ -z "${tflops}" && -n "${gflops}" ]]; then
      tflops=$(awk -v v="${gflops}" 'BEGIN{printf "%.3f", v/1000.0}')
      metric="TFLOPS=${tflops}"
    elif [[ -n "${tflops}" ]]; then
      metric="TFLOPS=${tflops}"
    fi
    if [[ "${metric}" == *"(GFLOPS="* ]]; then
      metric="${metric%% (GFLOPS=*}"
    fi
    if [[ -z "${metric}" ]]; then
      local gbps
      local gs
      local ms
      local special
      gbps="$(extract_gbps "${tmp}")"
      gs="$(extract_gsamples "${tmp}")"
      ms="$(extract_ms "${tmp}")"
      if [[ -n "${gbps}" ]]; then
        metric="GB/s=${gbps}"
        if [[ -n "${gs}" ]]; then
          metric="${metric} GSample/s=${gs}"
        fi
      elif [[ -n "${gs}" ]]; then
        metric="GSample/s=${gs}"
      elif [[ -n "${ms}" ]]; then
        metric="ms=${ms}"
      fi
      if [[ -z "${metric}" && ( "${label}" == *"rocSPARSE"* || "${label}" == *"hipSPARSE"* ) ]]; then
        special="$(extract_sparse_metrics "${tmp}")"
        metric="${special:-}"
      fi
      if [[ -z "${metric}" && "${label}" == *"rocRAND"* ]]; then
        special="$(extract_rocrand_metrics "${tmp}")"
        metric="${special:-}"
      fi
      if [[ -z "${metric}" && ( "${label}" == *"rocSOLVER"* || "${label}" == *"hipSOLVER"* ) ]]; then
        special="$(extract_single_number "${tmp}")"
        if [[ -n "${special}" ]]; then
          metric="gpu_time_us=${special}"
        fi
      fi
    fi
    add_result "${label}" "OK" "$(fmt_duration_ms "${elapsed_ms}")" "${metric}"
  else
    # Some upstream bench clients can print valid results but still exit non-zero
    # (observed: rocsolver-bench exits 255 while printing a full "Results" table).
    if [[ ${rc} -eq 255 ]] && [[ "${label}" == *"rocSOLVER"* ]] && rg -q "Results:|gpu_time" "${tmp}"; then
      add_result "${label}" "OK" "$(fmt_duration_ms "${elapsed_ms}")" "rc=255 (client exit-code bug; results printed)"
    else
      add_result "${label}" "FAIL" "$(fmt_duration_ms "${elapsed_ms}")" "rc=${rc}"
    fi
  fi
  rm -f "${tmp}"
  return ${rc}
}

run_bench_suite() {
  local timeout_s="$1"
  shift 1
  local -a selected=("$@")
  local expected_blas="${BENCH_EXPECTED}"
  local expected_misc="${BENCH_EXPECTED_MISC}"
  local qr_ops="" qr_bytes=""
  local sparse_ops="" sparse_bytes=""

  bench_selected() {
    local idx="$1"
    if (( ${#selected[@]} == 0 )); then
      return 0
    fi
    local s
    for s in "${selected[@]}"; do
      if [[ "${s}" == "${idx}" ]]; then
        return 0
      fi
    done
    return 1
  }

  # 1) BLAS (GEMM) — good "is my stack fast?" signal
  if bench_selected 1 && command -v rocblas-bench >/dev/null 2>&1; then
    BENCH_META_STATS=""
    # FLOP count: 2·m·n·k per GEMM. Approx. bytes: A(mk)+B(kn)+C(read+write,2mn).
    local gemm_ops gemm_bytes
    gemm_ops="$(awk -v n="${BENCH_SIZE}" -v it="${BENCH_ITERS}" 'BEGIN{printf "%.0f", 2.0*n*n*n*it}')"
    gemm_bytes="$(awk -v n="${BENCH_SIZE}" -v it="${BENCH_ITERS}" 'BEGIN{printf "%.0f", (n*n + n*n + 2.0*n*n)*4.0*it}')"
    BENCH_META_STATS="$(set_bench_meta_stats "${gemm_ops}" "FLOP" "${gemm_bytes}")"
    run_bench_with_timeout "bench: rocBLAS GEMM f32" "${expected_blas}" "${timeout_s}" \
      rocblas-bench -f gemm -r f32_r -m "${BENCH_SIZE}" -n "${BENCH_SIZE}" -k "${BENCH_SIZE}" \
      --alpha 1 --beta 0 --iters "${BENCH_ITERS}" || true
    BENCH_META_STATS=""
  elif bench_selected 1; then
    add_result "bench: rocBLAS GEMM f32" "SKIP" "0s" "rocblas-bench not in PATH (enable build.benchmarks=true, then rebuild rocBLAS)"
  fi

  if bench_selected 2 && command -v hipblas-bench >/dev/null 2>&1; then
    BENCH_META_STATS=""
    # If rocBLAS was skipped, compute GEMM stats here.
    local gemm_ops_h gemm_bytes_h
    gemm_ops_h="${gemm_ops:-}"
    gemm_bytes_h="${gemm_bytes:-}"
    if [[ -z "${gemm_ops_h}" || -z "${gemm_bytes_h}" ]]; then
      gemm_ops_h="$(awk -v n="${BENCH_SIZE}" -v it="${BENCH_ITERS}" 'BEGIN{printf "%.0f", 2.0*n*n*n*it}')"
      gemm_bytes_h="$(awk -v n="${BENCH_SIZE}" -v it="${BENCH_ITERS}" 'BEGIN{printf "%.0f", (n*n + n*n + 2.0*n*n)*4.0*it}')"
    fi
    BENCH_META_STATS="$(set_bench_meta_stats "${gemm_ops_h}" "FLOP" "${gemm_bytes_h}")"
    run_bench_with_timeout "bench: hipBLAS GEMM f32" "${expected_blas}" "${timeout_s}" \
      hipblas-bench -f gemm -r f32_r -m "${BENCH_SIZE}" -n "${BENCH_SIZE}" -k "${BENCH_SIZE}" \
      --alpha 1 --beta 0 --iters "${BENCH_ITERS}" || true
    BENCH_META_STATS=""
  elif bench_selected 2; then
    add_result "bench: hipBLAS GEMM f32" "SKIP" "0s" "hipblas-bench not in PATH (enable build.benchmarks=true, then rebuild hipBLAS)"
  fi

  if (( BENCH_LITE )); then
    return 0
  fi

  # 2) SOLVER (LAPACK-ish)
  if bench_selected 3 && command -v rocsolver-bench >/dev/null 2>&1; then
    # Use a sustained invocation so power/utilization sampling is meaningful.
    # Target: ~5s wall-time on gfx1031 (RX 6700 XT).
    local rocsolver_m rocsolver_batch rocsolver_iters
    if [[ "${MODE}" == "full" ]]; then
      # Double precision tends to be more compute-heavy and results in a more
      # sustained load than tiny single-precision runs.
      rocsolver_m="${ROC_SOLVER_M:-3072}"
      rocsolver_batch="${ROC_SOLVER_BATCH:-16}"
      rocsolver_iters="${ROC_SOLVER_ITERS:-1}"
    else
      rocsolver_m="${ROC_SOLVER_M:-2560}"
      rocsolver_batch="${ROC_SOLVER_BATCH:-17}"
      rocsolver_iters="${ROC_SOLVER_ITERS:-1}"
    fi
    local expected_solver
    if [[ "${MODE}" == "full" ]]; then
      expected_solver="typ. 5-10s"
    else
      expected_solver="typ. 4-7s"
    fi
    # rocsolver-bench can return NaNs for `-i 0`, so clamp to 1.
    if [[ "${rocsolver_iters}" == "0" ]]; then
      rocsolver_iters=1
    fi
    # Approx FLOPs for QR (square): (4/3)·n^3; data ~ 2·n^2 doubles.
    BENCH_META_STATS=""
    qr_ops="$(awk -v n="${rocsolver_m}" -v b="${rocsolver_batch}" -v it="${rocsolver_iters}" 'BEGIN{printf "%.0f", (4.0/3.0)*n*n*n*b*it}')"
    qr_bytes="$(awk -v n="${rocsolver_m}" -v b="${rocsolver_batch}" -v it="${rocsolver_iters}" 'BEGIN{printf "%.0f", 2.0*n*n*8.0*b*it}')"
    BENCH_META_STATS="$(set_bench_meta_stats "${qr_ops}" "FLOP" "${qr_bytes}")"
    run_bench_with_timeout "bench: rocSOLVER geqrf_strided_batched (d)" "${expected_solver}" "${timeout_s}" \
      rocsolver-bench -f geqrf_strided_batched -r d -m "${rocsolver_m}" --batch_count "${rocsolver_batch}" --perf 1 -i "${rocsolver_iters}" || true
    BENCH_META_STATS=""
  elif bench_selected 3; then
    add_result "bench: rocSOLVER geqrf_strided_batched (d)" "SKIP" "0s" "rocsolver-bench not in PATH"
  fi

  if bench_selected 4 && command -v hipsolver-bench >/dev/null 2>&1; then
    # Make it sustained so power/utilization sampling is meaningful.
    # Target: ~5s wall-time on gfx1031 (RX 6700 XT).
    local hipsolver_m hipsolver_iters
    if [[ "${MODE}" == "full" ]]; then
      hipsolver_m="${HIP_SOLVER_M:-10240}"
      hipsolver_iters="${HIP_SOLVER_ITERS:-1}"
    else
      hipsolver_m="${HIP_SOLVER_M:-8704}"
      hipsolver_iters="${HIP_SOLVER_ITERS:-1}"
    fi
    local expected_solver
    if [[ "${MODE}" == "full" ]]; then
      expected_solver="typ. 7-12s"
    else
      expected_solver="typ. 4-7s"
    fi
    # hipsolver-bench can return NaNs for `-i 0`, so clamp to 1.
    if [[ "${hipsolver_iters}" == "0" ]]; then
      hipsolver_iters=1
    fi
    # Use the same ops/data estimate as rocSOLVER geqrf_strided_batched (d),
    # even though the underlying math differs (requested for consistent reporting).
    BENCH_META_STATS=""
    if [[ -n "${qr_ops}" && -n "${qr_bytes}" ]]; then
      BENCH_META_STATS="$(set_bench_meta_stats "${qr_ops}" "FLOP" "${qr_bytes}")"
    fi
    run_bench_with_timeout "bench: hipSOLVER (tiny solver)" "${expected_solver}" "${timeout_s}" \
      hipsolver-bench --perf 1 -f getrf -r d -m "${hipsolver_m}" -n "${hipsolver_m}" -i "${hipsolver_iters}" || true
    BENCH_META_STATS=""
  elif bench_selected 4; then
    add_result "bench: hipSOLVER (tiny solver)" "SKIP" "0s" "hipsolver-bench not in PATH"
  fi

  # 3) SPARSE
  # Use the hipSPARSE axpyi parameters as the canonical ops/data estimate for both
  # rocSPARSE and hipSPARSE (requested for consistent reporting).
  if (bench_selected 5 || bench_selected 6) && [[ -z "${sparse_ops}" ]]; then
    local canon_n canon_nnz canon_iters
    if [[ "${MODE}" == "full" ]]; then
      canon_n="${HIPSPARSE_N:-2097152}"
      canon_nnz="${HIPSPARSE_NNZ:-524288}"
      canon_iters="${HIPSPARSE_ITERS:-115000}"
    else
      canon_n="${HIPSPARSE_N:-1048576}"
      canon_nnz="${HIPSPARSE_NNZ:-262144}"
      canon_iters="${HIPSPARSE_ITERS:-165000}"
    fi
    sparse_ops="$(awk -v nnz="${canon_nnz}" -v it="${canon_iters}" 'BEGIN{printf "%.0f", 2.0*nnz*it}')"
    sparse_bytes="$(awk -v nnz="${canon_nnz}" -v it="${canon_iters}" 'BEGIN{printf "%.0f", 28.0*nnz*it}')"
  fi

  if bench_selected 5 && command -v rocsparse-bench >/dev/null 2>&1; then
    # Sustained sparse workload to make power/utilization sampling meaningful.
    # Note: rocSPARSE uses `-m` for LEVEL-1 vector size in axpyi.
    local rocsparse_m rocsparse_nnz rocsparse_iters
    if [[ "${MODE}" == "full" ]]; then
      rocsparse_m="${ROCSPARSE_M:-2097152}"
      rocsparse_nnz="${ROCSPARSE_NNZ:-524288}"
      rocsparse_iters="${ROCSPARSE_ITERS:-90000}"
    else
      rocsparse_m="${ROCSPARSE_M:-1048576}"
      rocsparse_nnz="${ROCSPARSE_NNZ:-262144}"
      rocsparse_iters="${ROCSPARSE_ITERS:-120000}"
    fi
    local expected_sparse="typ. 4-7s"
    # AXPYI: ~2 FLOP per nnz (mul+add); data ~ (x val 8B + idx 4B + y read+write 16B)=28B per nnz.
    BENCH_META_STATS=""
    if [[ -n "${sparse_ops}" && -n "${sparse_bytes}" ]]; then
      BENCH_META_STATS="$(set_bench_meta_stats "${sparse_ops}" "FLOP" "${sparse_bytes}")"
    else
      BENCH_META_STATS=""
    fi
    run_bench_with_timeout "bench: rocSPARSE axpyi (d)" "${expected_sparse}" "${timeout_s}" \
      rocsparse-bench -f axpyi -r d -m "${rocsparse_m}" -z "${rocsparse_nnz}" -i "${rocsparse_iters}" --iters_inner 1 -v 0 || true
    BENCH_META_STATS=""
  elif bench_selected 5; then
    add_result "bench: rocSPARSE axpyi (d)" "SKIP" "0s" "rocsparse-bench not in PATH"
  fi

  if bench_selected 6 && command -v hipsparse-bench >/dev/null 2>&1; then
    # Sustained sparse workload to make power/utilization sampling meaningful.
    # Note: hipSPARSE uses `-n` for LEVEL-1 vector size in axpyi.
    local hipsparse_n hipsparse_nnz hipsparse_iters
    if [[ "${MODE}" == "full" ]]; then
      hipsparse_n="${HIPSPARSE_N:-2097152}"
      hipsparse_nnz="${HIPSPARSE_NNZ:-524288}"
      hipsparse_iters="${HIPSPARSE_ITERS:-115000}"
    else
      hipsparse_n="${HIPSPARSE_N:-1048576}"
      hipsparse_nnz="${HIPSPARSE_NNZ:-262144}"
      hipsparse_iters="${HIPSPARSE_ITERS:-165000}"
    fi
    local expected_sparse="typ. 4-7s"
    BENCH_META_STATS=""
    if [[ -n "${sparse_ops}" && -n "${sparse_bytes}" ]]; then
      BENCH_META_STATS="$(set_bench_meta_stats "${sparse_ops}" "FLOP" "${sparse_bytes}")"
    else
      BENCH_META_STATS=""
    fi
    run_bench_with_timeout "bench: hipSPARSE axpyi (d)" "${expected_sparse}" "${timeout_s}" \
      hipsparse-bench -f axpyi -r d -n "${hipsparse_n}" -z "${hipsparse_nnz}" -i "${hipsparse_iters}" --iters_inner 1 -v 0 || true
    BENCH_META_STATS=""
  elif bench_selected 6; then
    add_result "bench: hipSPARSE axpyi (d)" "SKIP" "0s" "hipsparse-bench not in PATH"
  fi

  # 4) FFT
  if bench_selected 7 && command -v rocfft-bench >/dev/null 2>&1; then
    # rocfft-bench reports per-sample timings and a single run is too short for
    # power/utilization sampling. Run a long profile silently, then a small one
    # to print metrics.
    local rocfft_len rocfft_batch rocfft_ntrial
    if [[ "${MODE}" == "full" ]]; then
      rocfft_len="${ROCFFT_LEN:-524288}"
      rocfft_batch="${ROCFFT_BATCH:-4}"
      rocfft_ntrial="${ROCFFT_NTRIAL:-3000}"
    else
      rocfft_len="${ROCFFT_LEN:-262144}"
      rocfft_batch="${ROCFFT_BATCH:-4}"
      rocfft_ntrial="${ROCFFT_NTRIAL:-8000}"
    fi
    local expected_fft="typ. 4-8s"
    # FFT: approx ~5·N·log2(N) complex-ops per transform. Data ~ 2·N complex64 (16B) per transform.
    BENCH_META_STATS=""
    local fft_ops fft_bytes
    fft_ops="$(awk -v n="${rocfft_len}" -v b="${rocfft_batch}" -v t="${rocfft_ntrial}" 'BEGIN{printf "%.0f", 5.0*n*(log(n)/log(2.0))*b*t}')"
    fft_bytes="$(awk -v n="${rocfft_len}" -v b="${rocfft_batch}" -v t="${rocfft_ntrial}" 'BEGIN{printf "%.0f", 2.0*n*16.0*b*t}')"
    BENCH_META_STATS="$(set_bench_meta_stats "${fft_ops}" "op" "${fft_bytes}")"
    run_bench_with_timeout "bench: rocFFT complex fwd (${rocfft_len}, batch=${rocfft_batch}, d)" "${expected_fft}" "${timeout_s}" \
      bash -lc "set -euo pipefail; rocfft-bench --length ${rocfft_len} --precision double -t 0 -b ${rocfft_batch} -N ${rocfft_ntrial} >/dev/null; rocfft-bench --length ${rocfft_len} --precision double -t 0 -b ${rocfft_batch} -N 2" || true
    BENCH_META_STATS=""
  elif bench_selected 7; then
    add_result "bench: rocFFT complex fwd (sustained)" "SKIP" "0s" "rocfft-bench not in PATH"
  fi

  if bench_selected 8 && command -v dyna-rocfft-bench >/dev/null 2>&1; then
    local lib
    lib="$(ls -1 "${ROCM_PATH}/lib/librocfft.so"* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${lib}" ]]; then
      local rocfft_len rocfft_batch rocfft_ntrial
      if [[ "${MODE}" == "full" ]]; then
        rocfft_len="${ROCFFT_LEN:-524288}"
        rocfft_batch="${ROCFFT_BATCH:-4}"
        rocfft_ntrial="${ROCFFT_NTRIAL:-3000}"
      else
        rocfft_len="${ROCFFT_LEN:-262144}"
        rocfft_batch="${ROCFFT_BATCH:-4}"
        rocfft_ntrial="${ROCFFT_NTRIAL:-8000}"
      fi
      local expected_fft="typ. 4-8s"
      BENCH_META_STATS=""
      local dfft_ops dfft_bytes
      dfft_ops="$(awk -v n="${rocfft_len}" -v b="${rocfft_batch}" -v t="${rocfft_ntrial}" 'BEGIN{printf "%.0f", 5.0*n*(log(n)/log(2.0))*b*t}')"
      dfft_bytes="$(awk -v n="${rocfft_len}" -v b="${rocfft_batch}" -v t="${rocfft_ntrial}" 'BEGIN{printf "%.0f", 2.0*n*16.0*b*t}')"
      BENCH_META_STATS="$(set_bench_meta_stats "${dfft_ops}" "op" "${dfft_bytes}")"
      run_bench_with_timeout "bench: dyna-rocFFT complex fwd (${rocfft_len}, batch=${rocfft_batch}, d)" "${expected_fft}" "${timeout_s}" \
        bash -lc "set -euo pipefail; dyna-rocfft-bench --lib '${lib}' --length ${rocfft_len} --precision double -t 0 -b ${rocfft_batch} -N ${rocfft_ntrial} >/dev/null; dyna-rocfft-bench --lib '${lib}' --length ${rocfft_len} --precision double -t 0 -b ${rocfft_batch} -N 2" || true
      BENCH_META_STATS=""
    else
      add_result "bench: dyna-rocFFT complex fwd (sustained)" "SKIP" "0s" "librocfft.so not found under ${ROCM_PATH}/lib"
    fi
  elif bench_selected 8; then
    add_result "bench: dyna-rocFFT complex fwd (sustained)" "SKIP" "0s" "dyna-rocfft-bench not in PATH"
  fi

  # 5) RNG
  if bench_selected 9 && command -v benchmark_rocrand_generate >/dev/null 2>&1; then
    # Sustained RNG workload to make power/utilization sampling meaningful.
    # The rocrand micro-bench reports an internal "Time(all)" that can be large,
    # but has low host overhead. Increase trials to get ~5s wall-time.
    local rocrand_size rocrand_trials
    if [[ "${MODE}" == "full" ]]; then
      rocrand_size="${ROCRAND_SIZE:-134217728}"
      rocrand_trials="${ROCRAND_TRIALS:-3300}"
    else
      rocrand_size="${ROCRAND_SIZE:-134217728}"
      rocrand_trials="${ROCRAND_TRIALS:-3300}"
    fi
    local expected_rng="typ. 4-7s"
    # RNG: count generated values and output bytes (float32).
    BENCH_META_STATS=""
    local rng_samples rng_bytes
    rng_samples="$(awk -v sz="${rocrand_size}" -v tr="${rocrand_trials}" 'BEGIN{printf "%.0f", 1.0*sz*tr}')"
    rng_bytes="$(awk -v sz="${rocrand_size}" -v tr="${rocrand_trials}" 'BEGIN{printf "%.0f", 1.0*sz*tr*4.0}')"
    BENCH_META_STATS="$(set_bench_meta_stats "${rng_samples}" "samples" "${rng_bytes}")"
    run_bench_with_timeout "bench: rocRAND generate (philox, uniform-float)" "${expected_rng}" "${timeout_s}" \
      benchmark_rocrand_generate --size "${rocrand_size}" --trials "${rocrand_trials}" --dis uniform-float --engine philox --format csv || true
    BENCH_META_STATS=""
  elif bench_selected 9; then
    add_result "bench: rocRAND generate (philox, uniform-float)" "SKIP" "0s" "benchmark_rocrand_generate not in PATH"
  fi
}

if [[ "${MODE}" == "full" ]]; then
  BENCH_SIZE="${BENCH_SIZE:-4096}"
  BENCH_ITERS="${BENCH_ITERS:-20}"
  # With power sampling enabled (default), GEMM is forced to ~5s. Without
  # power, GEMM is usually ~1-2s on RX 6700 XT (gfx1031).
  BENCH_EXPECTED="typ. 2-6s"
  BENCH_EXPECTED_MISC="typ. <1s"
  BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-900}"
else
  BENCH_SIZE="${BENCH_SIZE:-2048}"
  BENCH_ITERS="${BENCH_ITERS:-10}"
  # With power sampling enabled (default), GEMM is forced to ~5s. Without
  # power, GEMM is usually ~1-2s on RX 6700 XT (gfx1031).
  BENCH_EXPECTED="typ. 1-2s"
  BENCH_EXPECTED_MISC="typ. <1s"
  BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-300}"
fi

# If power sampling is enabled, prefer a sustained load so avgW/gpu% are meaningful.
# Only adjust if BENCH_ITERS is still at the default.
if (( RUN_POWER )); then
  BENCH_EXPECTED="typ. 4-7s"
  # For GEMM we want sustained load (~5s) to make power/gpu% sampling reliable.
  # Other bench clients ignore BENCH_ITERS, so this mainly affects rocBLAS/hipBLAS.
  if [[ "${MODE}" == "quick" && "${BENCH_ITERS}" == "10" ]]; then
    BENCH_ITERS="2400"
  elif [[ "${MODE}" == "full" && "${BENCH_ITERS}" == "20" ]]; then
    BENCH_ITERS="400"
  fi
fi

print_run_header "gfx1031 test run" "${BUILD_DIR}" "${ROCM_PATH}"

detect_expect_stage

if (( RUN_POWER )); then
  # Baseline is measured per-test inside power_wrap().
  discover_power_sensor || true
fi

if (( RUN_CONSISTENCY )); then
  echo "==== consistency checks (${BUILD_DIR}) ====" | tee -a "${LOG_FILE}"
  add_result "expected stage" "OK" "0s" "${EXPECT_STAGE:-unspecified}"
  check_cmd_available "tool present: ninja" ninja || true
  check_cmd_available "tool present: cmake" cmake || true
  check_cmd_available "tool present: ccache" ccache || true

  # Basic cache/path hygiene checks
  # Light scan excludes known-benign matches (packaging prefixes inside internal ExternalProject caches).
  check_no_opt_rocm_in_caches \
    "no /opt/rocm in caches" \
    "scan CMakeCache.txt under ${BUILD_DIR}" \
    "/compiler/amd-llvm/build/runtimes/|CPACK_PACKAGING_INSTALL_PREFIX:(STRING|PATH)=/opt/rocm|CMAKE_INSTALL_PREFIX:(STRING|PATH)=/opt/rocm|_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX:INTERNAL=/opt/rocm|FIND_PACKAGE_MESSAGE_DETAILS_HIP:INTERNAL=\\[/opt/rocm/bin\\]" || true
  check_toolchain_paths "${EXPECT_STAGE}" "toolchain" || true
  if (( HAVE_ROCM_ENV )) && [[ "${EXPECT_STAGE}" == "stage2" ]]; then
    check_hip_device_libs "hip" || true
  else
    add_result "hip device libs" "SKIP" "0s" "Stage-1 (or ROCM_PATH missing)"
  fi

  if (( CONSISTENCY_DEEP )); then
    # Deep scan: scan everything under BUILD_DIR, but still exclude known-benign
    # packaging defaults and internal caches that mention /opt/rocm without
    # actually *using* it as an effective search root.
    check_no_opt_rocm_in_caches \
      "no /opt/rocm in caches (deep)" \
      "full scan under ${BUILD_DIR} (excluding known-benign packaging defaults)" \
      "/compiler/amd-llvm/build/runtimes/|CPACK_PACKAGING_INSTALL_PREFIX:(STRING|PATH)=/opt/rocm|CMAKE_INSTALL_PREFIX:(STRING|PATH)=/opt/rocm|_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX:INTERNAL=/opt/rocm|FIND_PACKAGE_MESSAGE_DETAILS_HIP:INTERNAL=\\[/opt/rocm/bin\\]" || true
    if (( HAVE_ROCM_ENV )); then
      check_runtime_linkage "runtime" || true
    else
      add_result "runtime linkage" "SKIP" "0s" "ROCM_PATH missing"
    fi
  else
    add_result "runtime linkage" "SKIP" "0s" "use --deep to run ldd checks"
  fi
fi

if (( RUN_SANITY )); then
  if command -v rocminfo >/dev/null 2>&1; then
    run_timed "rocminfo" "typ. <1s" rocminfo
  else
    add_result "rocminfo" "SKIP" "0s" "not in PATH"
  fi

  if command -v hipinfo >/dev/null 2>&1; then
    run_timed "hipinfo" "typ. <1s" hipinfo
  else
    add_result "hipinfo" "SKIP" "0s" "not in PATH (build/install `core-hipinfo` to add it)"
  fi

  check_rocm_core_components "component" || true
fi

if (( RUN_CORE_LOAD )); then
  run_rocm_core_component_load_tests "component" || true
fi

if (( RUN_MIOPEN )); then
  run_miopen_checks "miopen" || true
fi

if (( RUN_BENCH )); then
  if (( BENCH_MENU )); then
    if [[ ! -t 0 ]]; then
      echo "ERROR: --bench-menu requires an interactive TTY (stdin)." | tee -a "${LOG_FILE}" >&2
      exit 2
    fi
    while true; do
      print_bench_menu | tee -a "${LOG_FILE}"
      read -r -p "Select bench test (0-12, list, q): " sel
      echo "Selection: ${sel}" | tee -a "${LOG_FILE}"
      case "${sel}" in
        q|quit|exit)
          break
          ;;
        l|list)
          continue
          ;;
        "")
          continue
          ;;
      esac
      # shellcheck disable=SC2207
      selected_arr=($(parse_bench_selection "${sel}"))
      # If user chose 0, run all (empty selection -> all).
      for s in "${selected_arr[@]}"; do
        if [[ "${s}" == "0" ]]; then
          selected_arr=()
          break
        fi
      done

      # Reset results per selection so the summary is for this run only.
      RESULT_LABELS=()
      RESULT_STATUS=()
      RESULT_TIME=()
      RESULT_METRIC=()

      run_bench_suite "${BENCH_TIMEOUT_S}" "${selected_arr[@]}"
      run_rocm_core_component_load_tests "component" "${selected_arr[@]}" || true

      echo "" | tee -a "${LOG_FILE}"
      print_summary_table "==== gfx1031 test summary ===="
      if (( LOG_ENABLED )); then
        echo "${C_DIM}Log:${C_RESET} ${LOG_FILE}" | tee -a "${LOG_FILE}"
      else
        echo "${C_DIM}Log:${C_RESET} (disabled; re-run with --log [file])" | tee -a "${LOG_FILE}"
      fi
    done
  else
    run_bench_suite "${BENCH_TIMEOUT_S}"
  fi
fi

echo "" | tee -a "${LOG_FILE}"
print_summary_table "==== gfx1031 test summary ===="

if (( LOG_ENABLED )); then
  echo "${C_DIM}Log:${C_RESET} ${LOG_FILE}" | tee -a "${LOG_FILE}"
else
  echo "${C_DIM}Log:${C_RESET} (disabled; re-run with --log [file])" | tee -a "${LOG_FILE}"
fi
