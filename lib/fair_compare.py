#!/usr/bin/env python3
"""
fair_compare.py — compare every backend this machine actually has, fairly.

The per-backend stages in the suite answer "does this backend work, and how fast
is it end to end". This answers the question you ask afterwards: "which backend
should I actually use, and by how much?" — which needs all of them measured
under the same conditions, at the same moment, with the spread reported.

It discovers backends rather than assuming them:

  PyTorch CPU                     always
  PyTorch GPU (eager)             torch.cuda.is_available()   — ROCm or CUDA
  PyTorch GPU (torch.compile)     same, plus a working Inductor
  PyTorch XPU (eager/compile)     torch.xpu.is_available()    — Intel
  ONNX Runtime <EP>               one row per execution provider that loads
  OpenVINO <device>               one row per ov.Core().available_devices
  RKNN NPU                        rknnlite present and an .rknn model beside it

Two rules it enforces, which is the point of the file:

  * A backend is named by what it ACTUALLY ran on, never by what was requested.
    ONNX rows report `sess.get_providers()[0]`; OpenVINO rows report the device
    the compiled model reports. A silent CPU fallback shows up as a CPU row.
  * Raw inference only — no NMS, no Ultralytics pre/post — so the comparison is
    between runtimes, not between their Python wrappers. The suite's end-to-end
    numbers live in the other stages.

Usage
-----
    python3 fair_compare.py --model yolo11n --workdir ~/yolo-amd-test
    python3 fair_compare.py --model yolo11n --mode interleaved
    python3 fair_compare.py --model yolo11n --csv out.csv
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fairness import Runner, measure, report, csv_rows  # noqa: E402

IMGSZ = 640


def _torch_runners(model_pt, x_np, want_compile=True):
    runners = []
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        print(f"  (torch/ultralytics unavailable: {e})")
        return runners

    # Deliberately NOT forcing a thread count. Runtimes differ in what is optimal
    # and forcing parity makes the comparison less useful, not more: on the AMD
    # ROCm torch build this was measured fastest at ONE thread (105 ms for
    # yolo26n) and 4x slower at 16 (448 ms), while ONNX Runtime wants every core.
    # Each backend gets its own best configuration, which is what a deployment
    # would do; the thread count is printed so the comparison stays legible.
    # Override with the usual environment variables if you want to explore it:
    #   OMP_NUM_THREADS=8 python3 fair_compare.py ...
    if os.environ.get("TORCH_NUM_THREADS"):
        try:
            torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))
        except Exception:
            pass

    def add(device, label, kind):
        try:
            net = YOLO(model_pt).model.to(device).eval()
            try:
                net = net.fuse()          # conv+bn fusion: what a real deployment does
            except Exception:
                pass
            x = torch.from_numpy(x_np).to(device)
            sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else \
                   (lambda: torch.xpu.synchronize()) if device == "xpu" else (lambda: None)

            def fn(net=net, x=x, sync=sync):
                with torch.inference_mode():
                    net(x)
                sync()
            runners.append(Runner(f"pytorch {label}", fn, kind=kind))

            if want_compile and device != "cpu":
                try:
                    cnet = torch.compile(net)

                    def cfn(cnet=cnet, x=x, sync=sync):
                        with torch.inference_mode():
                            cnet(x)
                        sync()
                    cfn()   # trigger compilation now, not inside the timer
                    runners.append(Runner(f"pytorch {label} (compiled)", cfn, kind=kind,
                                          note="graph-fused via torch.compile"))
                except Exception as e:
                    print(f"  (torch.compile on {label} unavailable: {str(e).splitlines()[0][:90]})")
        except Exception as e:
            print(f"  (pytorch {label} unavailable: {str(e).splitlines()[0][:90]})")

    add("cpu", "cpu", "cpu")
    if getattr(torch, "cuda", None) and torch.cuda.is_available():
        add("cuda", "gpu", "gpu")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        add("xpu", "xpu", "gpu")
    return runners


def _onnx_runners(model_onnx, x_np):
    runners = []
    if not os.path.exists(model_onnx):
        return runners
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"  (onnxruntime unavailable: {e})")
        return runners

    avail = ort.get_available_providers()
    print(f"  onnxruntime {ort.__version__}, providers: {avail}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    for ep in avail:
        if ep in ("AzureExecutionProvider", "TvmExecutionProvider"):
            continue
        try:
            sess = ort.InferenceSession(model_onnx, so, providers=[ep, "CPUExecutionProvider"])
        except Exception as e:
            print(f"  (ORT {ep} failed to create a session: {str(e).splitlines()[0][:90]})")
            continue
        actual = sess.get_providers()[0]     # what ran, not what was asked for
        if actual != ep and ep != "CPUExecutionProvider":
            print(f"  ORT {ep} fell back to {actual} — reporting it as {actual}")
        name = f"onnx {actual.replace('ExecutionProvider', '')}"
        if any(r.name == name for r in runners):
            continue
        nm = sess.get_inputs()[0].name
        kind = "cpu" if "CPU" in actual else ("npu" if any(k in actual for k in ("QNN", "VitisAI", "NPU")) else "gpu")
        runners.append(Runner(name, lambda s=sess, n=nm: s.run(None, {n: x_np}), kind=kind))
    return runners


def _openvino_runners(workdir, model, x_np):
    runners = []
    try:
        import openvino as ov
    except Exception:
        return runners
    core = ov.Core()
    for prec in ("fp16", "fp32"):
        d = os.path.join(workdir, f"{model}_{prec}_openvino_model")
        xml = os.path.join(d, f"{model}.xml")
        if not os.path.exists(xml):
            continue
        for dev in core.available_devices:
            try:
                cm = core.compile_model(xml, dev)
                real = cm.get_property("EXECUTION_DEVICES") if "EXECUTION_DEVICES" in cm.get_property_names() else dev
                real = real[0] if isinstance(real, (list, tuple)) and real else str(real)
                req = cm.create_infer_request()
                inp = cm.input(0)
                kind = "npu" if "NPU" in str(real).upper() else ("gpu" if "GPU" in str(real).upper() else "cpu")
                runners.append(Runner(f"openvino {real} {prec}",
                                      lambda r=req, i=inp: r.infer({i: x_np}), kind=kind))
            except Exception as e:
                print(f"  (openvino {dev} {prec} unavailable: {str(e).splitlines()[0][:90]})")
        break   # one precision is enough for a like-for-like comparison
    return runners


def _rknn_runners(workdir, model, x_np):
    runners = []
    path = os.path.join(workdir, f"{model}.rknn")
    if not os.path.exists(path):
        return runners
    try:
        from rknnlite.api import RKNNLite
    except Exception:
        return runners
    try:
        rk = RKNNLite()
        if rk.load_rknn(path) != 0 or rk.init_runtime() != 0:
            print("  (rknn: load/init failed)")
            return runners
        nhwc = np.transpose(x_np, (0, 2, 3, 1))
        runners.append(Runner("rknn npu", lambda: rk.inference(inputs=[nhwc]), kind="npu"))
    except Exception as e:
        print(f"  (rknn unavailable: {str(e).splitlines()[0][:90]})")
    return runners


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n", help="model stem, e.g. yolo11n or yolo26n")
    ap.add_argument("--workdir", default=".", help="directory holding the .pt/.onnx/IR files")
    ap.add_argument("--mode", default="isolated", choices=["isolated", "interleaved"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--no-compile", action="store_true", help="skip torch.compile rows")
    ap.add_argument("--csv", help="append rows to this CSV")
    args = ap.parse_args()

    wd = os.path.expanduser(args.workdir)
    os.chdir(wd)
    x = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)

    print(f"discovering backends for {args.model} in {wd} ...")
    runners = []
    runners += _torch_runners(f"{args.model}.pt", x, want_compile=not args.no_compile)
    runners += _onnx_runners(f"{args.model}.onnx", x)
    runners += _openvino_runners(wd, args.model, x)
    runners += _rknn_runners(wd, args.model, x)

    if not runners:
        print("no backends available — nothing to compare")
        return 1
    try:
        import torch as _t
        threads = f"; torch CPU threads = {_t.get_num_threads()} (set TORCH_NUM_THREADS to change)"
    except Exception:
        threads = ""
    print(f"comparing {len(runners)} backends in {args.mode} mode "
          f"({args.rounds} rounds x {args.reps} reps){threads}\n")

    tty = sys.stderr.isatty()
    res = measure(runners, mode=args.mode, rounds=args.rounds, reps=args.reps,
                  progress=(lambda m: print(f"  … {m}\033[K", end="\r", file=sys.stderr, flush=True)) if tty else None)
    if tty:
        print(" " * 78, end="\r", file=sys.stderr)
    report(res, mode=args.mode,
           title=f"{args.model} — raw inference, {IMGSZ}x{IMGSZ}, no NMS ({args.mode})")

    print("\n  These are raw runtime numbers. End-to-end FPS (decode + preprocess + NMS)")
    print("  is measured per backend elsewhere in the suite and will be lower.")
    print("  Prove the accelerator was really used:  python3 lib/accel_verify.py --list")

    if args.csv:
        with open(args.csv, "a") as f:
            for row in csv_rows(res, args.model):
                f.write(row + "\n")
        print(f"\n  appended {len(res) * 2} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
