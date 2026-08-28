# YOLO on AMD Ryzen iGPU/NPU — Diagnostic + Benchmark Suite

A full test suite for running [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO (YOLO11 and YOLO26) on AMD Ryzen APUs — covering the integrated GPU (via ROCm), CPU baseline, ONNX Runtime, and the NPU where available. It tells you what actually works on your specific chip, and how fast it runs, with real numbers instead of guesswork.

## Supported hardware

| iGPU | gfx target | Chip family (examples) |
|---|---|---|
| 780M | `gfx1103` | 7840HS / 8845HS / HX 255 (Phoenix) |
| 890M | `gfx1150` | HX 370/375/470/475 + PRO variants (Strix Point) |
| 880M | `gfx1150` | Ryzen AI 9 365/465 + PRO variants |
| 860M | `gfx1152` | Ryzen AI 7 350/450 + PRO variants |
| 840M | `gfx1152`/`gfx1153` | Ryzen AI 5 340/440, AI 7 345 + PRO variants |
| 8050S/8060S | `gfx1151` | Ryzen AI MAX 385/395 "Strix Halo" |

The script auto-detects your chip from `lspci` and tries the relevant `HSA_OVERRIDE_GFX_VERSION` combos, so it runs unmodified on any of the above.

## What it does

1. **System checks** — kernel version, `amdgpu` module, `/dev/kfd`, `render` group membership, `rocminfo`/`rocm-smi`, chip auto-detection.
2. **Environment setup** — creates an isolated venv (`~/yolo-amd-test/venv`), installs PyTorch (ROCm build), Ultralytics, `onnxruntime`, `onnxruntime-rocm`. Downloads:
   - A public test image (Ultralytics' `bus.jpg`)
   - Six models: **YOLO11** and **YOLO26**, nano/small/medium each
   - A real public test video — `solutions_ci_demo.mp4`, from [Ultralytics' official GitHub Releases CDN](https://github.com/ultralytics/assets/releases/tag/v0.0.0) (the same asset their own CI suite uses)
3. **GPU visibility matrix** — tests multiple `HSA_OVERRIDE_GFX_VERSION` / `HSA_ENABLE_SDMA` combinations against PyTorch, since the correct override varies by ROCm version and chip.
4. **Full benchmark (CPU + GPU/ROCm)** — for every model, every backend:
   - **Cold-load time** — `YOLO(model)` constructor cost (real app startup latency)
   - **First-inference time** — isolated separately, since GPU kernel compilation/JIT lives here and is usually much slower than steady state
   - **Steady-state latency** — avg / min / max / **p95** / stdev over 15 repeated runs
   - **Video FPS** — real end-to-end throughput (decode + preprocess + inference + NMS) over up to 300 frames of the test video
5. **ONNX Runtime + ROCm/MIGraphX backend** — same benchmark dimensions, tested for both YOLO11n and YOLO26n.
6. **NPU (XDNA) detection** — checks kernel version, `amdxdna` module, `xrt-smi`, and whether `onnxruntime` has the `VitisAIExecutionProvider`. Attempts inference through it if available. This stage is expected to **warn rather than fail** on most systems — see [NPU notes](#npu-notes) below.
7. **Summary** — pass/fail counts, three benchmark comparison tables (video FPS, steady-state latency, cold-load time) across every backend and model that worked, and an auto-generated env file with the exact exports that worked on your system.

## Requirements

- Ubuntu 24.04 (or similar)
- ROCm already installed at the system level (this script tests on top of it, it doesn't install ROCm/kernel drivers itself)
- Internet access to `github.com`, `pypi.org`, and `download.pytorch.org` for model/package downloads

If ROCm isn't installed yet, see the [ROCm setup steps](#one-time-rocm-setup) below.

## Usage

```bash
chmod +x test_yolo_amd.sh
./test_yolo_amd.sh                # full run: YOLO11 + YOLO26, n/s/m each, ~6 models
./test_yolo_amd.sh --quick        # nano only from each family — faster, less bandwidth
./test_yolo_amd.sh --skip-install # skip venv/package installs (reuse existing setup)
```

Safe to re-run at any time. It never touches your system Python — everything lives in `~/yolo-amd-test/venv`.

## Output

Everything is written under `~/yolo-amd-test/`:

| Path | Contents |
|---|---|
| `results_summary.txt` | Every PASS/FAIL/WARN line from the run |
| `benchmark_results.csv` | Raw numbers: `backend,model,stage,metric,value` — import into a spreadsheet or plot directly |
| `logs/` | Full stdout/stderr from every individual test, for debugging |
| `yolo_amd_env.sh` | Auto-generated exports that worked on your system |
| `bus.jpg`, `solutions_ci_demo.mp4`, `*.pt`, `*.onnx` | Downloaded test assets and models |

At the end of the run, the script prints three formatted tables (video FPS, steady-state ms/frame, cold-load ms) so you can compare CPU vs GPU vs ONNX, and YOLO11 vs YOLO26, at a glance.

### Using the results

The script asks whether to append the working exports to `~/.bashrc`. If you accept, `yolo predict device=0 ...` will just work in any new terminal afterward. If you decline (or run non-interactively), source the generated file manually before using YOLO:

```bash
source ~/yolo-amd-test/yolo_amd_env.sh
yolo predict model=yolo11n.pt source=your_image.jpg device=0
```

## NPU notes

AMD's NPU (XDNA) is **not** accessible through Ultralytics or PyTorch — there is no `device=npu`. The only path is ONNX Runtime + AMD's Vitis AI Execution Provider, part of the separate "Ryzen AI Software" stack.

- That stack is **Windows-first**. Official Linux support currently targets **Strix Point (890M-class, `gfx1150`) and later**, on Ubuntu 24.04 with kernel ≥6.10.
- If you're on a **780M (Phoenix)** system, NPU support on Linux is unofficial/unconfirmed — expect Step 6 to warn rather than pass. This is expected behavior, not a broken setup.
- Even where the EP is available, real deployment normally requires quantizing the model first with AMD's [Quark](https://ryzenai.docs.amd.com/) tool (INT8/BF16) — this script's NPU test runs a plain unquantized FP32 ONNX export, mainly to detect whether the EP is reachable at all.

## One-time ROCm setup

If ROCm isn't installed yet (run once, before this script):

```bash
# Kernel prerequisite for Ryzen APUs on Ubuntu 24.04
sudo apt update && sudo apt install linux-oem-24.04c
sudo reboot

# Headers + amdgpu-install package (ROCm 6.4.1)
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
wget https://repo.radeon.com/amdgpu-install/6.4.1/ubuntu/noble/amdgpu-install_6.4.60401-1_all.deb
sudo apt install ./amdgpu-install_6.4.60401-1_all.deb
sudo apt update

# Driver + ROCm + DKMS
sudo apt install python3-setuptools python3-wheel
sudo amdgpu-install -y --usecase=graphics,rocm,dkms,hip
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

After reboot, verify with `rocminfo` — you should see your GPU's `gfx` target listed as an agent.

## Files in this repo

- `test_yolo_amd.sh` — the main diagnostic + benchmark script
- `bench_harness.py` — reusable Python benchmark harness called by the script for each backend/model combination (cold-load, first-infer, steady-state, video FPS)
- `README.md` — this file

## Caveats

- GPU acceleration gains on these iGPUs are often modest for small models once NMS/pre/post overhead is counted — check the CPU baseline numbers in your own results before assuming GPU is faster.
- The `HSA_OVERRIDE_GFX_VERSION` trick is a compatibility workaround for chip targets PyTorch doesn't natively recognize. Newer ROCm releases increasingly need no override at all; the script tries `no_override` first for this reason.
- This script probes and benchmarks — it does not install ROCm, kernel packages, or NPU drivers. Those remain manual, one-time system-level steps.
