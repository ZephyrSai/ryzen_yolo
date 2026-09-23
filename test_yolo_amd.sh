#!/usr/bin/env bash
#
# test_yolo_amd.sh
#
# Full diagnostic + benchmark suite for running Ultralytics YOLO on AMD
# Ryzen iGPUs (and NPU, where available), covering the whole current
# APU lineage:
#
#   iGPU        gfx target   Chip family (examples)
#   ---------   ----------   -----------------------------------------
#   780M        gfx1103      7840HS / 8845HS / HX 255 (Phoenix)
#   890M        gfx1150      HX 370/375/470/475 + PRO variants (Strix Point)
#   880M        gfx1150      Ryzen AI 9 365/465 + PRO variants
#   860M        gfx1152      Ryzen AI 7 350/450 + PRO variants
#   840M        gfx1152/1153 Ryzen AI 5 340/440, AI 7 345 + PRO variants
#   8050S/8060S gfx1151      Ryzen AI MAX 385/395 "Strix Halo"
#
# Model coverage: both YOLO11 and YOLO26 (nano/small/medium of each, 6
# models total by default) are downloaded and benchmarked side by side,
# so you get a direct generation-over-generation comparison on your own
# hardware, not just per-backend numbers for one model family.
#
# The script auto-detects your gfx target from rocminfo/lspci and tries
# the override combos relevant to it, so the same script runs on any of
# the above without editing.
#
# Tests, in order:
#   0. System / driver sanity checks (kernel, amdgpu module, rocminfo)
#   1. Python env setup + download test image, 3 model sizes, and a real
#      public test video (Ultralytics' official CI demo asset)
#   2. PyTorch + ROCm GPU visibility, with and without HSA overrides
#   3. FULL BENCHMARK on CPU and GPU (PyTorch/ROCm), across all model sizes:
#        - cold-load time (first-ever model load/startup cost)
#        - first-inference time (GPU kernel compile/JIT included)
#        - steady-state image inference (avg/min/max/p95/stdev over 15 runs)
#        - real video FPS (decode+preprocess+inference+NMS, up to 300 frames)
#      All numbers written to benchmark_results.csv for easy comparison.
#   4. Same benchmark dimensions via ONNX Runtime + ROCm/MIGraphX backend
#   5. NPU (XDNA) detection + YOLO-on-NPU test via ONNX Runtime Vitis AI EP
#   6. Summary report: pass/fail table, benchmark comparison tables
#      (FPS / latency / cold-load across all backends+models), and
#      auto-generated env file with the exports that actually worked
#
# Usage:
#   chmod +x test_yolo_amd.sh
#   ./test_yolo_amd.sh                # run everything: YOLO11+YOLO26, n/s/m each
#   ./test_yolo_amd.sh --quick        # nano only from each family, faster/less bandwidth
#   ./test_yolo_amd.sh --skip-install # assume venv/deps already installed
#
# Safe to re-run. Does not touch system Python; creates an isolated venv
# at ~/yolo-amd-test/venv. Does NOT install ROCm/kernel/NPU driver packages
# itself — those are one-time system-level steps; this script assumes they
# are already installed (or absent) and just probes/tests what's there.
#
# NPU NOTES (read before expecting Step 5 to pass):
#   - AMD's NPU (XDNA) stack is accessed via ONNX Runtime's Vitis AI EP,
#     NOT via Ultralytics/PyTorch directly. There is no "device=npu" in
#     Ultralytics. This script tests the ONNX Runtime + Vitis AI EP path.
#   - This stack is Windows-first. Official Linux support ("Ryzen AI for
#     Linux") currently targets Strix Point (890M-class, gfx1150) and
#     later chips, on Ubuntu 24.04 with kernel >= 6.10. Phoenix-era 780M
#     NPU support on Linux is unconfirmed/unofficial.
#   - Real NPU deployment normally requires quantizing the model first
#     with AMD's Quark tool (INT8/BF16) — this script attempts a plain
#     FP32 ONNX run through the Vitis AI EP for detection purposes; a
#     WARN (not FAIL) on quantization-related errors is expected and
#     does not mean the NPU is unusable, just that the full quantized
#     flow needs to be set up separately.

set -uo pipefail

VENV_DIR="$HOME/yolo-amd-test/venv"
WORKDIR="$HOME/yolo-amd-test"
LOGDIR="$WORKDIR/logs"
RESULTS_FILE="$WORKDIR/results_summary.txt"
BENCH_CSV="$WORKDIR/benchmark_results.csv"
SKIP_INSTALL=0
QUICK_MODE=0

for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    --quick) QUICK_MODE=1 ;;
  esac
done

# Ultralytics "AutoUpdate" pip-installs anything its requirement check can't
# find. The ROCm/MIGraphX builds of onnxruntime are distributed under other
# distribution names, so the check fails and it silently installs the CPU
# onnxruntime over the GPU one — every later "GPU" ONNX result is then CPU.
export YOLO_AUTOINSTALL=False

mkdir -p "$WORKDIR" "$LOGDIR"
cp "$(dirname "$0")/bench_harness.py" "$WORKDIR/bench_harness.py"
: > "$RESULTS_FILE"
echo "backend,model,stage,metric,value_ms_or_fps" > "$BENCH_CSV"

# ---------- colors / helpers ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

section() { echo -e "\n${BLUE}==================================================================${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}==================================================================${NC}"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; echo "[PASS] $1" >> "$RESULTS_FILE"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; echo "[FAIL] $1" >> "$RESULTS_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[WARN] $1" >> "$RESULTS_FILE"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# ==================================================================
section "STEP 0: System / driver sanity checks"
# ==================================================================

echo "Kernel version:"
uname -r | tee "$LOGDIR/kernel_version.log"

echo -e "\nCPU model:"
lscpu | grep "Model name" | tee "$LOGDIR/cpu_model.log"

echo -e "\nChecking for amdgpu kernel module..."
if lsmod | grep -q amdgpu; then
  pass "amdgpu kernel module is loaded"
else
  fail "amdgpu kernel module NOT loaded — driver install likely incomplete or reboot needed"
fi

echo -e "\nChecking for /dev/kfd (ROCm compute device node)..."
if [ -e /dev/kfd ]; then
  pass "/dev/kfd exists (ROCm compute stack present)"
else
  fail "/dev/kfd missing — ROCm userspace/kernel driver not properly installed"
fi

echo -e "\nChecking render group membership for current user..."
if groups | grep -qE '\brender\b'; then
  pass "User is in 'render' group"
else
  warn "User NOT in 'render' group — run: sudo usermod -a -G render,video \$LOGNAME, then log out/in"
fi

echo -e "\nRunning rocminfo (GPU agent listing)..."
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo > "$LOGDIR/rocminfo_default.log" 2>&1
  if grep -q "gfx1103\|gfx1100\|gfx1101\|gfx1102" "$LOGDIR/rocminfo_default.log"; then
    GFX_TARGET=$(grep -oE "gfx[0-9a-f]+" "$LOGDIR/rocminfo_default.log" | grep -v gfx000 | sort -u | tail -1)
    pass "rocminfo detects a GPU agent: $GFX_TARGET (full log: $LOGDIR/rocminfo_default.log)"
  else
    warn "rocminfo ran but no recognizable RDNA3 gfx target found — see $LOGDIR/rocminfo_default.log"
  fi
else
  fail "rocminfo command not found — ROCm not installed or not on PATH"
fi

echo -e "\nChecking rocm-smi (utilization/monitoring tool)..."
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi > "$LOGDIR/rocm-smi.log" 2>&1
  pass "rocm-smi available (log: $LOGDIR/rocm-smi.log)"
else
  warn "rocm-smi not found (non-fatal, just means you can't easily watch GPU utilization)"
fi

echo -e "\nIdentifying iGPU model via lspci..."
GPU_PCI_STRING=$(lspci -nn 2>/dev/null | grep -iE "VGA|Display" | grep -i amd)
echo "$GPU_PCI_STRING" | tee "$LOGDIR/lspci_gpu.log"

# Best-effort chip -> gfx target mapping based on lspci string.
# This is informational (drives which override combos we prioritize in
# Step 2) — the real ground truth is whatever rocminfo reports once ROCm
# is up, which we already captured above.
DETECTED_CHIP="unknown"
DETECTED_GFX="unknown"
if echo "$GPU_PCI_STRING" | grep -qiE "1900|Phoenix"; then
  DETECTED_CHIP="780M (Phoenix)"; DETECTED_GFX="gfx1103"
elif echo "$GPU_PCI_STRING" | grep -qiE "150e|Strix Point|Radeon 890M|Radeon 880M"; then
  DETECTED_CHIP="890M/880M (Strix Point)"; DETECTED_GFX="gfx1150"
elif echo "$GPU_PCI_STRING" | grep -qiE "1586|Krackan|Radeon 860M|Radeon 840M"; then
  DETECTED_CHIP="860M/840M (Krackan Point)"; DETECTED_GFX="gfx1152"
elif echo "$GPU_PCI_STRING" | grep -qiE "1586.*Radeon 840M"; then
  DETECTED_CHIP="840M (Krackan Point, alt die)"; DETECTED_GFX="gfx1153"
elif echo "$GPU_PCI_STRING" | grep -qiE "150c|150d|Strix Halo|8050S|8060S"; then
  DETECTED_CHIP="8050S/8060S (Strix Halo)"; DETECTED_GFX="gfx1151"
fi

if [ "$DETECTED_CHIP" != "unknown" ]; then
  pass "Detected iGPU: $DETECTED_CHIP -> expected gfx target: $DETECTED_GFX"
else
  warn "Could not auto-identify iGPU model from lspci string. Will test all known override combos below regardless."
fi

# ==================================================================
section "STEP 1: Python environment setup"
# ==================================================================

if [ "$SKIP_INSTALL" -eq 0 ]; then
  if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip -q

  info "Installing PyTorch (ROCm 7.1 build)..."
  # Match the torch wheel to the system ROCm. ROCm >= 7.14 / 10.x ships
  # per-GPU wheels with native gfx1103/115x kernels (no HSA override needed);
  # older stacks fall back to the generic rocm wheels + HSA override.
  # ROCm <= 7.2 records its version in /opt/rocm/.info/version; the 7.14+/10.x
  # per-GPU packaging does not, so fall back to the amdrocm metapackage name.
  ROCM_VER=$(cut -d- -f1 /opt/rocm/.info/version 2>/dev/null)
  [ -z "$ROCM_VER" ] && ROCM_VER=$(dpkg-query -W -f='${Package}\n' 'amdrocm[0-9]*-gfx*' 2>/dev/null | sed -E 's/^amdrocm([0-9.]+)-gfx.*/\1/' | sort -V | tail -1)
  ROCM_MAJOR=${ROCM_VER%%.*}
  if [ -n "${TORCH_PIP_ARGS:-}" ]; then
    info "Installing PyTorch with TORCH_PIP_ARGS override: $TORCH_PIP_ARGS"
    # shellcheck disable=SC2086
    pip install -q $TORCH_PIP_ARGS
  elif [ "${ROCM_MAJOR:-0}" -ge 10 ]; then
    info "System ROCm $ROCM_VER -> torch 2.13.0+rocm10.0.0 with the device package for $DETECTED_GFX"
    pip install -q --index-url https://stable.repo.amd.com/rocm/whl-next/ \
      "torch[device-${DETECTED_GFX}]==2.13.0+rocm10.0.0" \
      "torchvision[device-${DETECTED_GFX}]==0.28.0+rocm10.0.0"
  else
    info "System ROCm ${ROCM_VER:-unknown} -> generic torch rocm7.1 wheels (gfx1103 needs HSA_OVERRIDE)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1 -q
  fi \
    && pass "PyTorch ROCm build installed ($(python3 -c "import torch;print(torch.__version__)" 2>/dev/null))" \
    || fail "PyTorch ROCm install failed — check network / index URL"

  info "Installing ultralytics (+ onnx/onnxslim, which its ONNX export needs and can no longer auto-install)..."
  pip install ultralytics "onnx>=1.12,<2" onnxslim -q \
    && pass "ultralytics installed" \
    || fail "ultralytics install failed"

  # Exactly ONE onnxruntime build may be present: the CPU, ROCm and MIGraphX
  # wheels all unpack into the same onnxruntime/ directory and the last one
  # installed wins. AMD publishes MIGraphX-EP wheels per ROCm release at
  # repo.radeon.com (ROCm 7.2.x and older); the ROCm EP was removed in ORT 1.23.
  # No GPU wheel exists for ROCm 7.14 / 10.x yet -> CPU only, reported as such.
  pip uninstall -y -q onnxruntime onnxruntime-rocm onnxruntime-migraphx >/dev/null 2>&1 || true
  ORT_GPU=0
  if [ -n "$ROCM_VER" ] && [ "${ROCM_MAJOR:-0}" -lt 10 ] && [ "${ROCM_MAJOR:-0}" -ge 6 ]; then
    ORT_INDEX="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VER%.*}/"
    info "Trying the AMD onnxruntime-migraphx wheel for ROCm ${ROCM_VER%.*} from $ORT_INDEX ..."
    if pip install -q onnxruntime-migraphx -f "$ORT_INDEX" 2>>"$LOGDIR/onnxruntime_rocm_install.log"; then
      pip install -q "numpy<2" >/dev/null 2>&1   # AMD's ORT wheels are built against numpy 1.x
      ORT_GPU=1; pass "onnxruntime-migraphx (GPU) installed for ROCm ${ROCM_VER%.*}"
    fi
  fi
  if [ "$ORT_GPU" -eq 0 ]; then
    pip install onnxruntime -q \
      && warn "No AMD GPU onnxruntime wheel for ROCm ${ROCM_VER:-?} — installed CPU onnxruntime; Step 4 measures ONNX on the CPU only." \
      || warn "onnxruntime CPU install failed"
  fi
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  info "Skipping installs (--skip-install passed), using existing venv"
fi

# Grab a small test image if we don't have one
TEST_IMG="$WORKDIR/bus.jpg"
if [ ! -f "$TEST_IMG" ]; then
  info "Downloading sample test image..."
  python3 - "$TEST_IMG" <<'PYEOF'
import sys, urllib.request
url = "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"
try:
    urllib.request.urlretrieve(url, sys.argv[1])
    print("downloaded ok")
except Exception as e:
    print(f"download failed: {e}")
PYEOF
fi

# Additional model sizes for the load-time / throughput sweep. Covers both
# YOLO11 and YOLO26 families (nano/small/medium each) so the benchmark
# tables directly compare the two generations. --quick trims to nano only
# from each family to save time/bandwidth — see arg parsing near top.
declare -a BENCH_MODEL_NAMES=("yolo11n" "yolo11s" "yolo11m" "yolo11l" "yolo11x" "yolo26n" "yolo26s" "yolo26m" "yolo26l" "yolo26x")
if [ "$QUICK_MODE" -eq 1 ]; then
  BENCH_MODEL_NAMES=("yolo11n" "yolo26n")
fi
for m in "${BENCH_MODEL_NAMES[@]}"; do
  MP="$WORKDIR/${m}.pt"
  if [ ! -f "$MP" ]; then
    info "Downloading ${m}.pt..."
    python3 - "$m" "$MP" <<'PYEOF'
import sys
from ultralytics import YOLO
YOLO(f"{sys.argv[1]}.pt")
PYEOF
    [ -f "${m}.pt" ] && mv -f "${m}.pt" "$MP" 2>/dev/null
  fi
done

# Nano weights from each family, used as the reference model for the
# ONNX/MIGraphX (Step 4) and NPU (Step 5) tests, which only test one
# model each (those paths are about proving the backend/EP works, not
# doing the full size sweep — the PyTorch/CPU benchmark in Step 3 already
# covers all sizes x both families).
MODEL_PT="$WORKDIR/yolo11n.pt"          # used by Step 4 (ONNX/MIGraphX)
MODEL_PT_26="$WORKDIR/yolo26n.pt"       # also available if you want to swap Step 4/5 to YOLO26

# Real public test video: Ultralytics' own official CI demo asset, hosted
# on their public GitHub Releases CDN (same source their test suite pulls
# from — a real MP4 with people/vehicles, not a placeholder).
#   https://github.com/ultralytics/assets/releases/tag/v0.0.0
TEST_VIDEO="$WORKDIR/solutions_ci_demo.mp4"
TEST_VIDEO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
if [ ! -f "$TEST_VIDEO" ]; then
  info "Downloading public test video (Ultralytics official demo asset, ~2-3MB)..."
  wget -q -O "$TEST_VIDEO" "$TEST_VIDEO_URL" \
    && pass "Test video downloaded: $TEST_VIDEO" \
    || fail "Test video download failed — check network access to github.com. URL: $TEST_VIDEO_URL"
fi
if [ -f "$TEST_VIDEO" ]; then
  VIDEO_INFO=$(python3 - "$TEST_VIDEO" <<'PYEOF'
import sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"frames={n} fps={fps:.1f} res={w}x{h}")
PYEOF
)
  info "Test video info: $VIDEO_INFO"
fi

# ==================================================================
section "STEP 2: PyTorch GPU visibility matrix (testing override combos)"
# ==================================================================
# We test multiple env var combinations because the "correct" override
# depends on your exact ROCm version and ends up being trial-and-error
# on unsupported/semi-supported iGPU targets.

# Real gfx target -> HSA_OVERRIDE_GFX_VERSION string used historically to
# spoof unsupported/newly-supported targets as an already-compiled RDNA3
# PyTorch target. As ROCm versions add native support for more of these
# (see ROCm 10.0 compatibility matrix), "no_override" increasingly just
# works on its own — hence it's always tried first.
#   gfx1103 (780M)        -> override 11.0.0
#   gfx1150 (890M/880M)   -> override 11.0.0 or 11.0.1
#   gfx1151 (Strix Halo)  -> override 11.0.0 (often unnecessary on recent ROCm)
#   gfx1152 (860M)        -> override 11.0.2
#   gfx1153 (840M alt)    -> override 11.0.2
declare -a OVERRIDE_LABELS=(
  "no_override"
  "gfx_11.0.0_no_sdma"
  "gfx_11.0.0"
  "gfx_11.0.1_no_sdma"
  "gfx_11.0.2_no_sdma"
  "gfx_11.0.2"
)
declare -a OVERRIDE_ENVS=(
  ""
  "HSA_OVERRIDE_GFX_VERSION=11.0.0 HSA_ENABLE_SDMA=0"
  "HSA_OVERRIDE_GFX_VERSION=11.0.0"
  "HSA_OVERRIDE_GFX_VERSION=11.0.1 HSA_ENABLE_SDMA=0"
  "HSA_OVERRIDE_GFX_VERSION=11.0.2 HSA_ENABLE_SDMA=0"
  "HSA_OVERRIDE_GFX_VERSION=11.0.2"
)

info "Detected chip suggests target $DETECTED_GFX — all combos below will still be tried since actual ROCm-version behavior varies."

WORKING_OVERRIDE=""
GPU_OK=0   # separate flag: "no_override" is a valid winner but its env string is empty

for i in "${!OVERRIDE_LABELS[@]}"; do
  LABEL="${OVERRIDE_LABELS[$i]}"
  ENVSET="${OVERRIDE_ENVS[$i]}"
  info "Testing combo: ${LABEL} (${ENVSET:-<none>})"

  OUT=$(env $ENVSET python3 - <<'PYEOF' 2>&1
import torch
try:
    avail = torch.cuda.is_available()
    print(f"CUDA_AVAILABLE={avail}")
    if avail:
        print(f"DEVICE_NAME={torch.cuda.get_device_name(0)}")
        x = torch.rand(1024, 1024, device="cuda:0")
        y = torch.rand(1024, 1024, device="cuda:0")
        z = (x @ y).sum().item()
        print(f"MATMUL_OK={z}")
except Exception as e:
    print(f"ERROR={e}")
PYEOF
)
  echo "$OUT" > "$LOGDIR/pytorch_gpu_${LABEL}.log"

  if echo "$OUT" | grep -q "CUDA_AVAILABLE=True" && echo "$OUT" | grep -q "MATMUL_OK="; then
    pass "PyTorch sees GPU and runs a matmul with [$LABEL] -> $(echo "$OUT" | grep DEVICE_NAME)"
    if [ "$GPU_OK" -eq 0 ]; then GPU_OK=1; WORKING_OVERRIDE="$ENVSET"; WORKING_LABEL="$LABEL"; fi
  elif echo "$OUT" | grep -q "CUDA_AVAILABLE=False"; then
    warn "PyTorch does NOT see GPU with [$LABEL] (falls back to CPU-only)"
  else
    fail "PyTorch/ROCm errored or crashed with [$LABEL] — see $LOGDIR/pytorch_gpu_${LABEL}.log"
  fi
done

if [ "$GPU_OK" -eq 1 ]; then
  pass "GPU works with [$WORKING_LABEL] (${WORKING_OVERRIDE:-no environment overrides needed}) — will use this for YOLO GPU tests below"
else
  warn "No override combo got PyTorch to see the GPU. GPU-based YOLO tests below will be skipped; CPU and (if available) ONNX/MIGraphX tests will still run."
fi

# ==================================================================
section "STEP 3: Full benchmark — CPU and GPU, all model sizes, image + video"
# ==================================================================
# For each backend (CPU always; GPU if Step 2 found a working override)
# and each downloaded model size, this runs bench_harness.py which reports:
#   - COLD_LOAD_MS     : YOLO(model) constructor time (first-ever load)
#   - FIRST_INFER_MS   : first .predict() call (GPU kernel compile/JIT lives here)
#   - STEADY_*_MS      : repeated-inference steady state (avg/min/max/p95/stdev)
#   - VIDEO_*          : real end-to-end FPS over the downloaded test video
#
# Results are written to $BENCH_CSV (backend,model,stage,metric,value) for
# easy comparison/plotting, and human-readable lines are echoed live.

run_benchmark() {
  local backend_label="$1"
  local device_arg="$2"
  local env_prefix="$3"   # e.g. "HSA_OVERRIDE_GFX_VERSION=11.0.0" or "" for none
  local model_name="$4"
  local model_path="$WORKDIR/${model_name}.pt"

  [ -f "$model_path" ] || { warn "Model $model_path missing, skipping ${backend_label}/${model_name}"; return; }

  info "Benchmarking [$backend_label] model=$model_name device=$device_arg ..."
  local logfile="$LOGDIR/bench_${backend_label}_${model_name}.log"
  local OUT
  OUT=$(env $env_prefix python3 "$WORKDIR/bench_harness.py" \
        --model "$model_path" \
        --device "$device_arg" \
        --image "$TEST_IMG" \
        --video "$TEST_VIDEO" \
        --backend-label "$backend_label" \
        --img-runs 15 \
        --video-max-frames 300 2>&1)
  echo "$OUT" > "$logfile"

  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    local cold first steady_avg steady_p95 vfps vms
    cold=$(echo "$OUT" | grep "^COLD_LOAD_MS=" | cut -d= -f2)
    first=$(echo "$OUT" | grep "^FIRST_INFER_MS=" | cut -d= -f2)
    steady_avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2)
    steady_p95=$(echo "$OUT" | grep "^STEADY_P95_MS=" | cut -d= -f2)
    vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
    vms=$(echo "$OUT" | grep "^VIDEO_AVG_MS=" | cut -d= -f2)

    pass "[$backend_label/$model_name] cold-load ${cold}ms | first-infer ${first}ms | steady-state ${steady_avg}ms avg (p95 ${steady_p95}ms) | video ${vfps:-N/A} FPS (${vms:-N/A}ms/frame)"

    {
      echo "$backend_label,$model_name,cold_load,ms,$cold"
      echo "$backend_label,$model_name,first_infer,ms,$first"
      echo "$backend_label,$model_name,steady_avg,ms,$steady_avg"
      echo "$backend_label,$model_name,steady_p95,ms,$steady_p95"
      [ -n "$vfps" ] && echo "$backend_label,$model_name,video,fps,$vfps"
      [ -n "$vms" ] && echo "$backend_label,$model_name,video,ms_per_frame,$vms"
    } >> "$BENCH_CSV"
  else
    fail "[$backend_label/$model_name] benchmark failed — see $logfile"
  fi
}

# --- CPU benchmarks (always run, all model sizes) ---
for m in "${BENCH_MODEL_NAMES[@]}"; do
  run_benchmark "cpu" "cpu" "" "$m"
done

# --- GPU benchmarks (only if Step 2 found a working combo) ---
if [ "$GPU_OK" -eq 1 ]; then
  for m in "${BENCH_MODEL_NAMES[@]}"; do
    run_benchmark "gpu_pytorch_rocm" "0" "$WORKING_OVERRIDE" "$m"
  done
else
  warn "Skipping GPU benchmarks — no working PyTorch/ROCm override found in Step 2"
fi

# ==================================================================
# ---------------------------------------------------------------------------
# STEP 3b: prove the GPU actually ran the work, and compare backends fairly.
#
# Every runtime here will silently fall back to the CPU when something is wrong
# while still reporting "GPU". accel_verify.py reads the kernel's own
# per-process DRM accounting, so the claim can be checked against the hardware.
# ---------------------------------------------------------------------------
section "STEP 3b: hardware verification + fair backend comparison"

LIBDIR="$(cd "$(dirname "$0")" && pwd)/lib"
cp -f "$LIBDIR"/accel_verify.py "$LIBDIR"/fairness.py "$LIBDIR"/fair_compare.py "$WORKDIR/" 2>/dev/null || true

if [ "$GPU_OK" -eq 1 ]; then
  info "Running GPU inference while watching the kernel's DRM counters..."
  ( cd "$WORKDIR" && env $WORKING_OVERRIDE python3 - <<'PYINNER' >/dev/null 2>&1 &
import time
import numpy as np
import torch
from ultralytics import YOLO
net = YOLO("yolo11n.pt").model.to("cuda").eval()
x = torch.from_numpy(np.random.rand(1, 3, 640, 640).astype(np.float32)).to("cuda")
end = time.time() + 14
with torch.inference_mode():
    while time.time() < end:
        net(x)
        torch.cuda.synchronize()
PYINNER
  ) &
  sleep 5
  INFER_PID=$(pgrep -n -f "python3 -$" 2>/dev/null || pgrep -n python3)
  if [ -n "$INFER_PID" ] && python3 "$WORKDIR/accel_verify.py" --pid "$INFER_PID" --seconds 5 > "$LOGDIR/accel_verify.log" 2>&1; then
    pass "GPU verified against kernel counters: $(grep -oE 'compute [0-9.]+%|enc [0-9.]+%' "$LOGDIR/accel_verify.log" | paste -sd' ' -)"
  else
    warn "Could not prove GPU activity from kernel counters (see $LOGDIR/accel_verify.log). PyTorch reported a GPU but no DRM engine showed work - treat the GPU numbers as unproven."
  fi
  wait 2>/dev/null || true
else
  info "Skipping hardware verification - no working GPU backend found in Step 2"
fi

info "Fair backend comparison (isolated, interleaved rounds, spread reported)..."
if ( cd "$WORKDIR" && env $WORKING_OVERRIDE python3 fair_compare.py --model yolo11n --workdir "$WORKDIR" --csv "$BENCH_CSV" > "$LOGDIR/fair_compare.log" 2>&1 ); then
  pass "Fair comparison complete (full table in $LOGDIR/fair_compare.log)"
  sed -n '/^backend/,/^$/p' "$LOGDIR/fair_compare.log" | head -12
else
  warn "Fair comparison failed - see $LOGDIR/fair_compare.log"
fi

section "STEP 4: YOLO inference — ONNX Runtime + ROCm/MIGraphX backend"
# ==================================================================

python3 -c "import onnxruntime" >/dev/null 2>&1
ORT_INSTALLED=$?

if [ $ORT_INSTALLED -ne 0 ]; then
  warn "onnxruntime not installed — skipping ONNX test entirely"
else
  info "Checking available ONNX Runtime execution providers..."
  python3 -c "import onnxruntime as ort; print(ort.get_available_providers())" > "$LOGDIR/ort_providers.log" 2>&1
  cat "$LOGDIR/ort_providers.log"

  if grep -q "ROCMExecutionProvider\|MIGraphXExecutionProvider" "$LOGDIR/ort_providers.log"; then
    pass "ROCm/MIGraphX execution provider is available to ONNX Runtime"
  else
    warn "No ROCm/MIGraphX execution provider found in onnxruntime — only CPUExecutionProvider available. This path needs a ROCm-specific onnxruntime wheel matching your installed ROCm version (see repo.radeon.com)."
  fi

  # Test both YOLO11 and YOLO26 nano models through this backend for a
  # direct generation comparison (YOLO26 is NMS-free by default and is
  # reported by Ultralytics to be faster on CPU ONNX than YOLO11n).
  for onnx_model_name in "yolo11n" "yolo26n"; do
    onnx_src_pt="$WORKDIR/${onnx_model_name}.pt"
    MODEL_ONNX="$WORKDIR/${onnx_model_name}.onnx"
    [ -f "$onnx_src_pt" ] || { warn "$onnx_src_pt missing, skipping ONNX test for $onnx_model_name"; continue; }

    info "Exporting ${onnx_model_name} to ONNX..."
    python3 - "$onnx_model_name" "$onnx_src_pt" <<'PYEOF' > "$LOGDIR/onnx_export_${onnx_model_name}.log" 2>&1
import sys
from ultralytics import YOLO
model = YOLO(sys.argv[2])
model.export(format="onnx")
PYEOF
    [ -f "$WORKDIR/${onnx_model_name}.onnx" ] || find "$WORKDIR" -maxdepth 1 -name "${onnx_model_name}.onnx" -exec mv {} "$MODEL_ONNX" \;

    if [ ! -f "$MODEL_ONNX" ]; then
      fail "ONNX export failed for $onnx_model_name — see $LOGDIR/onnx_export_${onnx_model_name}.log"
      continue
    fi
    pass "ONNX export succeeded: $MODEL_ONNX"

    # Run regardless of which EP exists — the result line names the EP that
    # really executed, so a CPU-only onnxruntime is reported as CPU, never as GPU.
    info "Running ONNX inference for ${onnx_model_name} (best available EP; raw tensor timing + video FPS)..."
    OUT=$(python3 - "$MODEL_ONNX" "$TEST_VIDEO" <<'PYEOF' 2>&1
import sys
import time
import statistics
import numpy as np
import cv2
import onnxruntime as ort

model_onnx, video_path = sys.argv[1], sys.argv[2]
providers = ort.get_available_providers()
ep = "MIGraphXExecutionProvider" if "MIGraphXExecutionProvider" in providers else "ROCMExecutionProvider"

so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(model_onnx, so, providers=[ep, "CPUExecutionProvider"])
ep = sess.get_providers()[0]   # what actually ran, not what was asked for
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
shape = [d if isinstance(d, int) else 1 for d in input_shape]
if shape[0] == 1 and len(shape) == 4:
    shape = [1, 3, 640, 640]
dummy = np.random.rand(*shape).astype(np.float32)

t0 = time.perf_counter()
sess.run(None, {input_name: dummy})  # first run / warmup (includes any lazy compile)
first_ms = (time.perf_counter() - t0) * 1000

times = []
for _ in range(15):
    t0 = time.perf_counter()
    sess.run(None, {input_name: dummy})
    times.append((time.perf_counter() - t0) * 1000)

print(f"EP_USED={ep}")
print(f"FIRST_INFER_MS={first_ms:.2f}")
print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
print(f"STEADY_MIN_MS={min(times):.2f}")
print(f"STEADY_MAX_MS={max(times):.2f}")

# Real video pass (decode + resize + raw tensor inference; note: no NMS,
# this ONNX raw-output path doesn't decode boxes, this is throughput only)
h, w = shape[2], shape[3]
cap = cv2.VideoCapture(video_path)
n = 0
frame_times = []
t_start = time.perf_counter()
while n < 300:
    ok, frame = cap.read()
    if not ok:
        break
    resized = cv2.resize(frame, (w, h))
    tensor = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    t0 = time.perf_counter()
    sess.run(None, {input_name: tensor})
    frame_times.append((time.perf_counter() - t0) * 1000)
    n += 1
t_total = time.perf_counter() - t_start
cap.release()
if n > 0:
    print(f"VIDEO_FRAMES={n}")
    print(f"VIDEO_AVG_FPS={n / t_total:.2f}")
    print(f"VIDEO_AVG_MS={statistics.mean(frame_times):.2f}")
PYEOF
)
    echo "$OUT" > "$LOGDIR/yolo_onnx_rocm_${onnx_model_name}.log"

    if echo "$OUT" | grep -q "STEADY_AVG_MS="; then
      AVG=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2)
      FIRST=$(echo "$OUT" | grep "^FIRST_INFER_MS=" | cut -d= -f2)
      EP=$(echo "$OUT" | grep "^EP_USED=" | cut -d= -f2)
      VFPS=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
      if [ "$EP" = "CPUExecutionProvider" ]; then
        warn "ONNX Runtime ran on CPU (${EP}) [$onnx_model_name] — first-infer ${FIRST}ms | steady-state ${AVG}ms/frame | video ${VFPS:-N/A} FPS (raw tensor, no NMS). No GPU execution provider for this ROCm."
      else
        pass "ONNX Runtime GPU (${EP}) [$onnx_model_name] — first-infer ${FIRST}ms | steady-state ${AVG}ms/frame | video ${VFPS:-N/A} FPS (raw tensor, no NMS)"
      fi
      {
        echo "onnx_${EP},${onnx_model_name},first_infer,ms,$FIRST"
        echo "onnx_${EP},${onnx_model_name},steady_avg,ms,$AVG"
        [ -n "$VFPS" ] && echo "onnx_${EP},${onnx_model_name},video,fps,$VFPS"
      } >> "$BENCH_CSV"
    else
      fail "ONNX Runtime inference failed for $onnx_model_name — see $LOGDIR/yolo_onnx_rocm_${onnx_model_name}.log"
    fi
  done
fi

# ==================================================================
section "STEP 5: NPU (XDNA) detection + Vitis AI Execution Provider test"
# ==================================================================
# IMPORTANT CONTEXT (see header notes):
#  - Ultralytics/PyTorch have no NPU device — the only path to the NPU is
#    ONNX Runtime + AMD's Vitis AI Execution Provider (part of "Ryzen AI
#    Software for Linux"), which is a *separate install* from ROCm.
#  - Official Linux NPU support currently targets Strix Point (890M-class,
#    gfx1150) and later on Ubuntu 24.04 w/ kernel >=6.10. If you're on a
#    780M (Phoenix) system, expect this whole stage to likely report
#    "not available" — that's expected, not a bug in this script.
#  - We only PROBE here (driver presence, XRT tools, EP availability).
#    We do not install the NPU driver .deb packages or Ryzen AI Software
#    itself — that's a manual one-time step (see summary notes at the end).

echo "Checking kernel version against NPU minimum (>=6.10)..."
KVER=$(uname -r | grep -oE '^[0-9]+\.[0-9]+')
KVER_MAJOR=$(echo "$KVER" | cut -d. -f1)
KVER_MINOR=$(echo "$KVER" | cut -d. -f2)
if [ "$KVER_MAJOR" -gt 6 ] || { [ "$KVER_MAJOR" -eq 6 ] && [ "$KVER_MINOR" -ge 10 ]; }; then
  pass "Kernel $KVER meets the >=6.10 minimum for amdxdna NPU driver"
else
  warn "Kernel $KVER is below 6.10 — amdxdna NPU driver requires 6.10+. Consider: sudo apt install linux-generic-hwe-24.04"
fi

echo -e "\nChecking for amdxdna kernel module..."
if lsmod | grep -q amdxdna; then
  pass "amdxdna kernel module is loaded"
elif [ -e /sys/module/amdxdna ]; then
  pass "amdxdna module present in sysfs"
else
  warn "amdxdna kernel module NOT loaded — NPU driver not installed, or your kernel doesn't include it, or your chip predates official Linux NPU support (Phoenix/780M-era is unconfirmed)"
fi

echo -e "\nChecking for XRT tools (xrt-smi) — installed separately via AMD's Ryzen AI Linux driver package..."
if command -v xrt-smi >/dev/null 2>&1; then
  pass "xrt-smi found"
  info "Running: xrt-smi examine"
  xrt-smi examine > "$LOGDIR/xrt-smi.log" 2>&1
  cat "$LOGDIR/xrt-smi.log"
  if grep -qi "NPU" "$LOGDIR/xrt-smi.log"; then
    pass "NPU device enumerated by xrt-smi (see $LOGDIR/xrt-smi.log)"
  else
    warn "xrt-smi ran but no NPU device listed — see $LOGDIR/xrt-smi.log"
  fi
else
  warn "xrt-smi not found — Ryzen AI Software / XRT NPU driver package not installed. Install guide: https://ryzenai.docs.amd.com/en/latest/linux.html (Ubuntu 24.04, kernel >=6.10, STX/KRK platforms only)"
fi

echo -e "\nChecking whether onnxruntime has the Vitis AI Execution Provider..."
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())" > "$LOGDIR/ort_providers_npu_check.log" 2>&1
cat "$LOGDIR/ort_providers_npu_check.log"

if grep -q "VitisAIExecutionProvider" "$LOGDIR/ort_providers_npu_check.log"; then
  pass "VitisAIExecutionProvider is available to ONNX Runtime"

  info "Attempting a plain FP32 ONNX inference through the Vitis AI EP (unquantized — real NPU deployment normally needs Quark INT8/BF16 quantization first, so a failure/fallback here is common and non-fatal)..."
  MODEL_ONNX="$WORKDIR/yolo11n.onnx"
  if [ -f "$MODEL_ONNX" ]; then
    OUT=$(python3 - <<PYEOF 2>&1
import time
import numpy as np
import onnxruntime as ort

try:
    sess = ort.InferenceSession(
        "$MODEL_ONNX",
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

    sess.run(None, {input_name: dummy})  # warmup

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    actual_providers = sess.get_providers()
    print(f"ACTUAL_PROVIDERS={actual_providers}")
    print(f"AVG_MS={sum(times)/len(times):.2f}")
except Exception as e:
    print(f"ERROR={e}")
PYEOF
)
    echo "$OUT" > "$LOGDIR/yolo_npu_vitisai.log"

    if echo "$OUT" | grep -q "AVG_MS=" && echo "$OUT" | grep "ACTUAL_PROVIDERS" | grep -q "VitisAI"; then
      AVG=$(echo "$OUT" | grep AVG_MS | cut -d= -f2)
      pass "NPU (Vitis AI EP) inference ran — avg ${AVG}ms/frame (unquantized FP32; quantized INT8/BF16 via Quark would be faster — see summary)"
      echo "npu_vitisai,yolo11n,steady_avg,ms,$AVG" >> "$BENCH_CSV"
    elif echo "$OUT" | grep -q "AVG_MS="; then
      warn "Session created but silently fell back to CPUExecutionProvider (no actual NPU execution) — see $LOGDIR/yolo_npu_vitisai.log"
    else
      warn "Vitis AI EP inference failed (expected if model isn't quantized, or opset/op support mismatch) — see $LOGDIR/yolo_npu_vitisai.log. This does not mean the NPU is broken, just that this quick unquantized test didn't run on it."
    fi
  else
    warn "No ONNX model available from Step 4 to test (ONNX export may have failed earlier) — skipping NPU inference test"
  fi
else
  warn "VitisAIExecutionProvider not available in onnxruntime. To get it: install AMD's Ryzen AI Software for Linux (https://ryzenai.docs.amd.com/en/latest/linux.html), which provides a Vitis-AI-EP-enabled onnxruntime build. Standard pip onnxruntime/onnxruntime-rocm builds do NOT include this EP."
fi

# ==================================================================
section "SUMMARY"
# ==================================================================

echo -e "\nFull results log: $RESULTS_FILE"
echo -e "Per-test logs in: $LOGDIR"
echo -e "Full benchmark numbers (CSV): $BENCH_CSV\n"

PASS_COUNT=$(grep -c "^\[PASS\]" "$RESULTS_FILE" || true)
FAIL_COUNT=$(grep -c "^\[FAIL\]" "$RESULTS_FILE" || true)
WARN_COUNT=$(grep -c "^\[WARN\]" "$RESULTS_FILE" || true)

echo -e "${GREEN}Passed: $PASS_COUNT${NC}  ${YELLOW}Warnings: $WARN_COUNT${NC}  ${RED}Failed: $FAIL_COUNT${NC}\n"

cat "$RESULTS_FILE"

echo -e "\n${BLUE}--------------------------------------------------------------${NC}"
echo -e "${BLUE}  BENCHMARK TABLE (video FPS, all backends/models — higher is better)${NC}"
echo -e "${BLUE}--------------------------------------------------------------${NC}"
printf "%-22s %-10s %10s\n" "BACKEND" "MODEL" "VIDEO_FPS"
grep ",video,fps," "$BENCH_CSV" | while IFS=, read -r backend model stage metric value; do
  printf "%-22s %-10s %10s\n" "$backend" "$model" "$value"
done
echo -e "${BLUE}--------------------------------------------------------------${NC}"
echo -e "${BLUE}  STEADY-STATE LATENCY (single image, ms/frame — lower is better)${NC}"
echo -e "${BLUE}--------------------------------------------------------------${NC}"
printf "%-22s %-10s %10s\n" "BACKEND" "MODEL" "MS/FRAME"
grep ",steady_avg,ms," "$BENCH_CSV" | while IFS=, read -r backend model stage metric value; do
  printf "%-22s %-10s %10s\n" "$backend" "$model" "$value"
done
echo -e "${BLUE}--------------------------------------------------------------${NC}"
echo -e "${BLUE}  COLD LOAD TIME (first-ever model load, ms — one-time app startup cost)${NC}"
echo -e "${BLUE}--------------------------------------------------------------${NC}"
printf "%-22s %-10s %10s\n" "BACKEND" "MODEL" "MS"
grep ",cold_load,ms," "$BENCH_CSV" | while IFS=, read -r backend model stage metric value; do
  printf "%-22s %-10s %10s\n" "$backend" "$model" "$value"
done

echo -e "\n${BLUE}--------------------------------------------------------------${NC}"

ENV_FILE="$WORKDIR/yolo_amd_env.sh"
: > "$ENV_FILE"
echo "# Auto-generated by test_yolo_amd.sh on $(date)" >> "$ENV_FILE"
echo "# Source this before running YOLO: source $ENV_FILE" >> "$ENV_FILE"

if [ "$GPU_OK" -eq 1 ] && grep -q "gpu_pytorch_rocm/" "$RESULTS_FILE"; then
  echo -e "${GREEN}RECOMMENDATION:${NC} Use PyTorch/ROCm backend."
  if [ -z "$WORKING_OVERRIDE" ]; then
    echo "  No HSA_OVERRIDE_GFX_VERSION needed — this ROCm/torch build has native kernels for your GPU."
    echo "# No HSA overrides needed: native GPU support in this ROCm/torch build." >> "$ENV_FILE"
  else
    echo "Required exports (written to $ENV_FILE):"
  fi
  for kv in $WORKING_OVERRIDE; do
    echo "  export $kv"
    echo "export $kv" >> "$ENV_FILE"
  done
  echo "export ULTRALYTICS_DEVICE=0  # use with: yolo predict device=\$ULTRALYTICS_DEVICE model=... source=..." >> "$ENV_FILE"

elif grep -q "ONNX Runtime GPU.*—.*first-infer" "$RESULTS_FILE"; then
  echo -e "${GREEN}RECOMMENDATION:${NC} PyTorch/ROCm path failed but ONNX Runtime + ROCm/MIGraphX worked."
  echo "Export your models to ONNX and run inference through onnxruntime with the ROCm/MIGraphX execution provider."
  EP_LINE=$(grep "EP_USED=" "$LOGDIR/yolo_onnx_rocm.log" 2>/dev/null | cut -d= -f2)
  echo "Required exports (written to $ENV_FILE):"
  if [ -n "$WORKING_OVERRIDE" ]; then
    for kv in $WORKING_OVERRIDE; do
      echo "  export $kv"
      echo "export $kv" >> "$ENV_FILE"
    done
  fi
  echo "  # Execution provider to request in onnxruntime.InferenceSession(providers=[...]): ${EP_LINE:-ROCMExecutionProvider}"
  echo "# ONNX Runtime execution provider for this session: providers=[\"${EP_LINE:-ROCMExecutionProvider}\", \"CPUExecutionProvider\"]" >> "$ENV_FILE"

else
  echo -e "${YELLOW}RECOMMENDATION:${NC} GPU acceleration did not work on this system via either backend."
  echo "Compare the CPU baseline numbers above against your throughput needs — iGPU gains for"
  echo "small models are often marginal (or negative) once NMS/pre/post overhead is factored in, so CPU-only"
  echo "may already be acceptable depending on your model size and frame rate needs."
  echo "No GPU exports needed — run with device=\"cpu\"." >> "$ENV_FILE"
fi

echo ""
if grep -q "NPU (Vitis AI EP) inference ran" "$RESULTS_FILE"; then
  echo -e "${GREEN}NPU:${NC} Vitis AI EP ran successfully on this system (unquantized). For real deployment,"
  echo "quantize your model with AMD's Quark tool (INT8/BF16) first — this typically gives a large speedup"
  echo "and is what AMD's own YOLO benchmarks use. Guide: https://ryzenai.docs.amd.com/en/latest/"
  {
    echo ""
    echo "# NPU (Vitis AI EP) — required at the RUNTIME level, not just Python env:"
    echo "export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
    echo "source /opt/xilinx/xrt/setup.sh"
    echo "# ONNX Runtime providers list to request: [\"VitisAIExecutionProvider\", \"CPUExecutionProvider\"]"
  } >> "$ENV_FILE"
elif grep -q "VitisAIExecutionProvider is available" "$RESULTS_FILE"; then
  echo -e "${YELLOW}NPU:${NC} Vitis AI EP is installed but the quick unquantized test didn't run on it cleanly."
  echo "This is expected — proper NPU deployment needs Quark quantization first, not a bare ONNX export."
  {
    echo ""
    echo "# NPU (Vitis AI EP) present but needs a quantized (Quark INT8/BF16) model to run cleanly:"
    echo "export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH"
    echo "source /opt/xilinx/xrt/setup.sh"
  } >> "$ENV_FILE"
else
  echo -e "${YELLOW}NPU:${NC} Not usable on this system as configured. Ultralytics has no NPU backend regardless —"
  echo "NPU access always requires the separate ONNX Runtime + Vitis AI EP stack (AMD 'Ryzen AI Software'),"
  echo "which on Linux currently officially supports Strix Point (890M/gfx1150-class) chips and later, on"
  echo "Ubuntu 24.04 with kernel >=6.10. If you're on a 780M (Phoenix) system, NPU support is unofficial/"
  echo "unconfirmed at this time — the iGPU (ROCm) or CPU paths above are the reliable options."
fi
echo -e "${BLUE}--------------------------------------------------------------${NC}\n"

echo -e "${BLUE}Env file written to: $ENV_FILE${NC}"
echo "Contents:"
echo "---"
cat "$ENV_FILE"
echo "---"

echo -e "\nTo use these every session, either:"
echo "  A) source it manually each time:   source $ENV_FILE"
echo "  B) make it permanent by appending to ~/.bashrc"
echo ""
if [ -t 0 ]; then
  read -r -p "Append the required export lines to ~/.bashrc now? [y/N] " APPEND_CHOICE
else
  APPEND_CHOICE="n"
  info "Non-interactive shell detected — skipping the ~/.bashrc prompt automatically. Run 'source $ENV_FILE' manually, or re-run this script in an interactive terminal to be asked."
fi
if [[ "$APPEND_CHOICE" =~ ^[Yy]$ ]]; then
  {
    echo ""
    echo "# --- Added by test_yolo_amd.sh on $(date) ---"
    grep '^export' "$ENV_FILE"
    echo "# --- end test_yolo_amd.sh block ---"
  } >> "$HOME/.bashrc"
  pass "Appended export lines to ~/.bashrc — restart your shell or run: source ~/.bashrc"
else
  info "Skipped. Run 'source $ENV_FILE' manually before using YOLO, or append it to ~/.bashrc / your shell profile yourself later."
fi
