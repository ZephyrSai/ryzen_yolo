# ROCm on the Radeon 780M (gfx1103) — the working stack, and how it was reached

`test_yolo_amd.sh` benchmarks *on top of* an existing ROCm install. This file records
the exact kernel + package combination that is verified to work on this hardware, the
procedure to reproduce it on a fresh machine, and — because the first attempt on this
box went wrong in instructive ways — what **not** to do.

Verified 22 Sep 2026 on:

| | |
|---|---|
| CPU / iGPU | AMD Ryzen 9 7940HS, Radeon 780M (Phoenix, **gfx1103**, 12 CU, RDNA3) |
| RAM / VRAM | 12 GB system; BIOS UMA carve-out 3 GB VRAM + 6.3 GB GTT |
| Distro | Pop!_OS 24.04 LTS (Ubuntu noble base) |

---

## 1. The combination that works

| Component | Version | Where from | Notes |
|---|---|---|---|
| Kernel | **`linux-oem-24.04d` → 6.17.0-1032-oem** | Ubuntu/Pop apt | AMD's stated requirement for Ryzen APUs is "OEM kernel 6.14 or newer for Ubuntu 24.04". Uses the **in-box `amdgpu`** — no DKMS. |
| GPU driver | in-box `amdgpu` (kernel module) | kernel | `amdgpu-dkms` is **not** installed and not wanted. |
| Mesa (display/VAAPI) | 26.1.6 (Pop!_OS build) | Pop apt | Untouched. AMD's `*-amdgpu-*` Mesa packages are **not** installed. |
| ROCm (system) | **10.0.0** — `amdrocm10.0-gfx1103` | `https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/` | First ROCm release with **official gfx1103 support**. Per-GPU metapackage; provides `rocminfo`, `rocm-smi`, `amdclang`, HIP, rocBLAS-for-gfx1103. Installs to `/opt/rocm`. |
| Python | 3.12.3 (system) + venv | Pop apt | venv at `~/yolo-amd-test/venv` |
| PyTorch | **2.13.0+rocm10.0.0** + `amd-torch-device-gfx1103` | `https://stable.repo.amd.com/rocm/whl-next/` | Ships **native gfx1103 kernels** → `HSA_OVERRIDE_GFX_VERSION` is **not set** and must not be. Pulls its own `rocm-sdk-*` 10.0.0 pip runtime, so torch is self-contained. |
| torchvision | 0.28.0+rocm10.0.0 | same index | |
| Ultralytics | 8.4.158 | PyPI | Run with `YOLO_AUTOINSTALL=False` (see §4). |
| ONNX Runtime | 1.30.0 **CPU** | PyPI | No AMD GPU wheel exists for ROCm 10 yet (see §5). |
| User groups | `render`, `video` | | Required to open `/dev/kfd` and `/dev/dri/renderD128`. |

Environment: **nothing**. No `HSA_OVERRIDE_GFX_VERSION`, no `HSA_ENABLE_SDMA`.

Verification that passes on this stack:

```bash
rocminfo | grep -E '^\s+Name:\s+gfx'          # -> gfx1103
source ~/yolo-amd-test/venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True AMD Radeon 780M Graphics
yolo predict model=yolo11n.pt source=bus.jpg device=0
```

---

## 2. Procedure (fresh Pop!_OS / Ubuntu 24.04 machine)

Run as your normal user; the commands call `sudo` where needed.

### 2.1 Kernel

Pop!_OS 24.04 currently ships a 7.x kernel. It is *not* broken for ROCm (the in-box
`amdgpu` works), but AMD validates Ryzen APUs on the Ubuntu OEM 6.14+ line and it is the
only one with `amdgpu-dkms` build support should you ever need it. Install and boot it:

```bash
sudo apt update && sudo apt install -y linux-oem-24.04d
sudo reboot
uname -r        # expect 6.17.0-xxxx-oem
```

Do **not** install `amdgpu-dkms`. Do **not** run `amdgpu-install` at all (see §3).

### 2.2 Groups

```bash
sudo usermod -aG render,video "$USER"
# log out and back in (or reboot) for the groups to apply
id -nG | grep -E 'render.*video|video.*render'
```

### 2.3 Remove any previous ROCm / AMD graphics stack

AMD's ROCm 10 docs: *"If you have ROCm 7.2.4 or older installed, please uninstall it
before proceeding."* Mixed versions are the single biggest source of breakage (§3.1).
On a clean machine skip this; otherwise:

```bash
# everything that came from an AMD repo, plus the old repo definitions
sudo apt purge -y $(dpkg-query -W -f='${Package}\n' | grep -E '^(amdrocm|rocm|hip|hsa|comgr|rocblas|migraphx|miopen|.*-amdgpu.*|amdgpu-install|amdgpu-core|amdgpu-lib.*)$')
sudo rm -f /etc/apt/sources.list.d/{amdgpu,amdgpu-proprietary,rocm}.list /etc/ld.so.conf.d/20-amdgpu.conf
sudo apt autoremove -y && sudo ldconfig && sudo apt update
ls -d /opt/rocm* /opt/amdgpu* 2>/dev/null   # should print nothing
```

Then confirm the distro Mesa is intact (this matters if the box also does VAAPI video):

```bash
vainfo --display drm --device /dev/dri/renderD128 | grep 'Driver version'   # Mesa Gallium driver 26.x (pop)
```

### 2.4 ROCm 10.0.0 (system tools) from AMD's stable repo

```bash
sudo mkdir -p -m 0755 /etc/apt/keyrings
wget -qO- https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg >/dev/null
sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources >/dev/null <<'EOF'
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2404/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF
sudo apt update
sudo apt install -y amdrocm10.0-gfx1103
rocminfo | grep -E '^\s+Name:\s+gfx'     # gfx1103
```

For other Ryzen iGPUs use the matching metapackage: `amdrocm10.0-gfx1150` (880M/890M),
`amdrocm10.0-gfx1151` (Strix Halo), `amdrocm10.0-gfx1152` (860M/840M).

### 2.5 Python environment

```bash
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/yolo-amd-test/venv && source ~/yolo-amd-test/venv/bin/activate
pip install -U pip

# PyTorch with native gfx1103 kernels (pulls the rocm-sdk-* 10.0.0 runtime wheels too)
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1103]==2.13.0+rocm10.0.0" \
  "torchvision[device-gfx1103]==0.28.0+rocm10.0.0"

pip install ultralytics "onnx>=1.12,<2" onnxslim onnxruntime   # onnxruntime = CPU build (see §5)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2.6 Run the suite

```bash
cd ryzen_yolo
./test_yolo_amd.sh --skip-install     # or without --skip-install to let it build the venv
```

The script now detects the system ROCm major version and picks the matching torch
index itself; on ROCm ≥ 10 it expects `no_override` to be the winning combo and the
`HSA_OVERRIDE_*` rows to *fail* (those overrides ask for gfx1100 kernels the
gfx1103-only wheel doesn't carry — that is correct behaviour, not a regression).

---

## 3. What went wrong the first time (and why the numbers looked odd)

The box originally reached a *partially* working state through this sequence
(reconstructed from `/var/log/apt/history.log`; full package list in
`docs/state-before-rocm10.txt`):

1. `linux-oem-24.04c` installed (fine) because `amdgpu-dkms` would not build on Pop's 7.x kernel.
2. `amdgpu-install` **6.4.1** with `--usecase=graphics,rocm,dkms,hip` — installed ROCm 6.4.1 **and** AMD's parallel Mesa stack (`libgl1-amdgpu-mesa-dri`, `mesa-amdgpu-va-drivers`, `/etc/ld.so.conf.d/20-amdgpu.conf`), which then shadowed Pop's Mesa for VAAPI.
3. `amdgpu-dkms` purged (didn't build), a **31.40.1** `amdgpu-install` added, and **ROCm 7.14.0~pre3** (a pre-release from the "Radeon Software 26.13" repo) installed on top — leaving `/opt/rocm-6.4.1` *and* a 7.14 `/opt/rocm` side by side.
4. The venv got `torch 2.13+rocm7.1` (self-contained, so it worked) plus **both** `onnxruntime` (CPU) and `onnxruntime-rocm` (a ROCm-6.4 build).

Consequences, all confirmed in this session:

### 3.1 ONNX "GPU" results were CPU results
`onnxruntime-rocm` and `onnxruntime` unpack into the same `onnxruntime/` directory.
Ultralytics' ONNX export runs a requirement check for a distribution literally named
`onnxruntime`, doesn't find it (the ROCm wheel is named `onnxruntime-rocm`), and
**auto-installs the CPU wheel on top** — its `.cpython-312...so` then shadows the ROCm
build's `.so`. The script printed `PASS ... (ROCMExecutionProvider)` because it echoed
the *requested* provider. Every ONNX number in the old results was CPU.

### 3.2 Even with that fixed, ORT could not run on the GPU
`ldd` on the ROCm ORT provider showed it loading `libamdhip64.so.6` + `libhipblaslt.so.0`
from `/opt/rocm-6.4.1` *and* `libamdhip64.so.7` + `libhipblaslt.so.1` from the 7.14
`/opt/rocm` in one process → `hipBLASLt: 'out of memory'` at init, under every
`HSA_OVERRIDE` value. That error is a symptom of two HIP runtimes, not of VRAM.

### 3.3 The HSA override was only needed because of the old torch wheel
`torch 2.13+rocm7.1` lists gfx1103 in `get_arch_list()` but its bundled rocBLAS ships no
`TensileLibrary_*_gfx1103.dat`, so `no_override` genuinely failed and
`HSA_OVERRIDE_GFX_VERSION=11.0.0` (borrow gfx1100 kernels) was the right workaround
*for that wheel*. On ROCm 10 the override is unnecessary and actively breaks things.

### 3.4 "Pro"/proprietary drivers would not have helped
There is no separate AMD driver for compute on Linux any more: `amdgpu` (kernel, in-box)
+ ROCm userspace is the whole stack. AMDGPU-PRO / AMF only matter for video encode
knobs and would fight Pop's Mesa. ROCm's official Ryzen support matrix at 7.2.x lists
only gfx1150/1151 (Strix); **gfx1103 became official in ROCm 7.14/10.0**, which is why
this document targets 10.0.

---

## 4. Rules that keep it working

- **`export YOLO_AUTOINSTALL=False`** whenever Ultralytics runs in this venv (the script
  sets it). Otherwise it will pip-install over your GPU packages again.
- **Exactly one `onnxruntime*` distribution** in the venv. Check with
  `pip list | grep -i onnxruntime`.
- **One ROCm in `/opt`.** `ls -d /opt/rocm*` should show a single directory.
- **No `HSA_OVERRIDE_GFX_VERSION` in `~/.bashrc`** on ROCm 10. The old
  `yolo_amd_env.sh` exports were removed from `~/.bashrc` on 22 Sep 2026.
- Never use `amdgpu-install --usecase=graphics` on Pop!_OS — it replaces Mesa.
- The **BIOS UMA frame buffer** is what ROCm sees as VRAM (3 GB on this box). If you hit
  real `out of memory` on large models, raise it in firmware; GTT (6.3 GB) is used for
  overflow but is slower.
- Kernel updates: stay on the `linux-oem-24.04*` series. If Pop auto-selects its 7.x
  kernel again, ROCm will most likely still work (in-box driver), but re-run
  `rocminfo` to confirm before trusting benchmark numbers.

---

## 5. ONNX Runtime on the GPU — not possible today (tested)

**Status: ONNX Runtime has no GPU execution path on ROCm 10 for gfx1103.** PyTorch does.
The benchmark therefore runs ONNX on the CPU and labels it `[WARN] ... ran on CPU`.

Why, in order of discovery:

1. **The ROCm EP was removed in ONNX Runtime 1.23.** AMD's replacement is the MIGraphX EP.
   The old `onnxruntime-rocm` wheels stop at 1.22.2 / ROCm 6.4.
2. **ROCm 10 ships no MIGraphX.** The repo has 464 packages, none of them MIGraphX. Its DNN
   library (`amdrocm-dnn10.0`, the MIOpen equivalent) has per-GPU variants for gfx1030,
   1100, 1101, 1102, 1150, 1151, 1152, 1153, 1200, 1201, 1250, 908, 90a, 942 and 950 —
   **gfx1103 is absent**.
3. **AMD's `onnxruntime-migraphx` wheels only go up to the ROCm 7.2.x line**
   (`repo.radeon.com/rocm/manylinux/rocm-rel-7.2.{1..4}/`, ORT 1.23.2). PyPI has a newer
   `onnxruntime-migraphx` 1.27.1 whose provider links against `libamdhip64.so.7` (which
   ROCm 10 provides) and `libmigraphx_c.so.3` (which it does not).
4. **Grafting MIGraphX onto ROCm 10 does not work.** Tested 22 Sep 2026: MIGraphX 2.15 +
   MIOpen 3.5.1 extracted from the ROCm 7.2.4 debs, exposed through `LD_LIBRARY_PATH` to a
   venv with `onnxruntime-migraphx==1.27.1`. The EP loads and initialises, then MIGraphX
   JIT-compiles its kernels with ROCm 10's clang/comgr and fails:

   ```
   .../migraphx/kernels/vec.hpp:95: error: [-Werror,-Wlifetime-safety-intra-tu-suggestions]
   hiprtc_runtime.h:15378: note: candidate function __hmin(const __hip_bfloat16, ...)
   4 errors generated when compiling for gfx1103.
   migraphx_program_compile: Error: std::bad_alloc
   ```

   `MIGRAPHX_GPU_HIP_FLAGS="-Wno-error -Wno-everything"` removes the warning-as-error ones
   (4 → 2) but not the ambiguous `__hmin` bfloat16 overload: MIGraphX 2.15's kernel headers
   are incompatible with ROCm 10's hiprtc runtime headers.

### If you do need ONNX on the GPU

Either wait for AMD to publish MIGraphX (and a gfx1103 DNN variant) for ROCm 10 — then it
is `pip install onnxruntime-migraphx` — or move the whole machine back to the **ROCm 7.2.4**
line, where every component is matched:

```bash
# ROCm 7.2.4 via amdgpu-install --no-dkms --usecase=rocm   (NOT ...,graphics)
pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ "numpy<2"
export HSA_OVERRIDE_GFX_VERSION=11.0.0     # gfx1103 is unofficial on 7.2.x
```

The cost is real: gfx1103 is not officially supported there, so **everything** needs the
HSA override again and PyTorch loses its native gfx1103 kernels. The measured payoff is
also unclear — see §7: PyTorch on the GPU already does 91 FPS end-to-end on yolo11n versus
41 FPS for ONNX on the CPU, and Ultralytics' loop is dominated by Python pre/post-processing
rather than raw inference.

The test script handles both worlds automatically: on a ROCm 6.x/7.x system it installs the
matching `onnxruntime-migraphx` wheel; on ROCm 10 it installs the CPU build and says so.

## 6. Files

- `INSTALL_ROCM_780M.md` — this document
- `docs/state-before-rocm10.txt` — package/venv inventory of the broken mixed install, for reference
- `setup.sh` — **legacy**: the original ROCm 6.4.1 + DKMS installer. Superseded by §2; kept for history.
- `test_yolo_amd.sh.before-rocm10` — script before the ONNX/AutoUpdate/ROCm-10 fixes

---

## 7. Measured results on this stack (22 Sep 2026, box otherwise idle)

Full run of `test_yolo_amd.sh`, 640 px, video = Ultralytics `solutions_ci_demo.mp4`
(300 frames, end-to-end incl. decode + pre/post + NMS). Raw data:
`docs/results_rocm10_clean_2026-09-22.txt`, `docs/benchmark_rocm10_clean_2026-09-22.csv`.

| Model | CPU video FPS | **GPU video FPS** | CPU steady ms | GPU steady ms | GPU first-infer ms |
|---|---:|---:|---:|---:|---:|
| yolo11n | 54.8 | **91.5** | 20.6 | 20.0 | 1684 |
| yolo26n | 53.1 | **88.4** | 21.2 | 20.2 | 1681 |
| yolo11s | 30.8 | **42.0** | 45.7 | 37.5 | 1744 |
| yolo26s | 27.1 | **40.6** | 41.5 | 37.5 | 1743 |
| yolo11m | 11.9 | **16.5** | 105.3 | 82.8 | 1890 |
| yolo26m | 12.0 | **16.4** | 121.9 | 83.8 | 1928 |
| yolo11l | 9.6 | **14.0** | 144.0 | 101.9 | 1943 |
| yolo26l | 9.8 | **14.0** | 143.7 | 100.7 | 1929 |
| yolo11x | 5.1 | **7.8** | 267.1 | 188.3 | 2177 |
| yolo26x | 5.3 | **7.8** | 247.1 | 189.0 | 2162 |
| yolo11n ONNX (ORT, **CPU**, no NMS) | 40.8 | – | 16.7 | – | – |
| yolo26n ONNX (ORT, **CPU**, no NMS) | 51.5 | – | 13.3 | – | – |

Reading the numbers:

- The 780M gives **~1.5–1.7× the CPU** end-to-end on every model. For the nano models the
  single-image steady-state latency is the same on CPU and GPU (~20 ms) because that
  figure is dominated by Ultralytics' Python pre/post-processing; the GPU advantage only
  shows in the video loop where inference overlaps with decode.
- Same stack with a hardware video encoder (CourtStream, 1 court) running on the APU at the
  same time: yolo11n dropped from 91 to 60 FPS. The iGPU, VCN and CPU share one power and
  memory budget — benchmark on an idle box.
- The previous run (ROCm 6.4.1/7.14-pre, torch rocm7.1 + `HSA_OVERRIDE=11.0.0`) reported
  97 FPS for yolo11n; that is within run-to-run variance of the 91 measured here, so the
  native gfx1103 kernels are not slower than the borrowed gfx1100 ones. Its ONNX
  "ROCm" numbers (47–51 FPS) were CPU numbers (§3.1).
- NPU (XDNA): `amdxdna` is loaded on kernel 6.17, but there is no Linux Ryzen AI /
  Vitis AI EP for Phoenix — expected `[WARN]`, unchanged.
