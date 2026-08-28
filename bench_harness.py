#!/usr/bin/env python3
"""
bench_harness.py — reusable benchmark harness called by test_yolo_amd.sh
for every backend/device/model combination.

Usage:
  python3 bench_harness.py --model <path.pt> --device <cpu|0> \
      --image <path.jpg> --video <path.mp4> --backend-label <str> \
      [--img-runs 15] [--video-max-frames 300]

Prints machine-readable KEY=VALUE lines (one per line) so the calling
bash script can grep them out. Measures, in order:
  1. COLD_LOAD_MS      - time to construct YOLO(model_path) fresh (import
                          + weight load + device placement), i.e. what a
                          real application pays once at startup.
  2. FIRST_INFER_MS    - time for the very first .predict() call, which on
                          GPU backends includes kernel compilation/JIT and
                          is usually much slower than steady state.
  3. STEADY_AVG_MS / STEADY_MIN_MS / STEADY_MAX_MS / STEADY_P95_MS
                        - N repeated inferences on the same image after
                          warmup, i.e. true steady-state per-frame cost.
  4. VIDEO_FRAMES, VIDEO_TOTAL_S, VIDEO_AVG_FPS, VIDEO_AVG_MS,
     VIDEO_MIN_FPS, VIDEO_MAX_FPS
                        - full video decode+inference loop, capped at
                          --video-max-frames frames, reporting real
                          end-to-end FPS (decode + preprocess + inference
                          + NMS), which is the number that actually
                          matters for real-world usage.
"""

import argparse
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True, help="'cpu' or '0' etc.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--video", required=False, default=None)
    ap.add_argument("--backend-label", required=True)
    ap.add_argument("--img-runs", type=int, default=15)
    ap.add_argument("--video-max-frames", type=int, default=300)
    args = ap.parse_args()

    device = args.device
    # ultralytics wants an int-like string for GPU indices, "cpu" for CPU
    if device != "cpu":
        try:
            device = int(device)
        except ValueError:
            pass

    print(f"BACKEND={args.backend_label}")
    print(f"MODEL={args.model}")
    print(f"DEVICE={device}")

    # ---------- 1. Cold load ----------
    t0 = time.perf_counter()
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
    except Exception as e:
        print(f"COLD_LOAD_ERROR={e}")
        sys.exit(1)
    cold_load_ms = (time.perf_counter() - t0) * 1000
    print(f"COLD_LOAD_MS={cold_load_ms:.1f}")

    # ---------- 2. First inference (includes GPU kernel compile / JIT) ----------
    t0 = time.perf_counter()
    try:
        results = model.predict(args.image, device=device, verbose=False)
    except Exception as e:
        print(f"FIRST_INFER_ERROR={e}")
        sys.exit(1)
    first_infer_ms = (time.perf_counter() - t0) * 1000
    print(f"FIRST_INFER_MS={first_infer_ms:.1f}")
    print(f"FIRST_RUN_DETECTIONS={len(results[0].boxes)}")

    # ---------- 3. Steady-state image inference ----------
    times = []
    for _ in range(args.img_runs):
        t0 = time.perf_counter()
        results = model.predict(args.image, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    times_sorted = sorted(times)
    p95_idx = max(0, int(len(times_sorted) * 0.95) - 1)
    print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
    print(f"STEADY_MIN_MS={min(times):.2f}")
    print(f"STEADY_MAX_MS={max(times):.2f}")
    print(f"STEADY_P95_MS={times_sorted[p95_idx]:.2f}")
    print(f"STEADY_STDEV_MS={statistics.pstdev(times):.2f}")
    print(f"STEADY_DETECTIONS={len(results[0].boxes)}")

    # ---------- 4. Full video benchmark (real end-to-end FPS) ----------
    if args.video:
        try:
            import cv2
        except Exception as e:
            print(f"VIDEO_ERROR=opencv not available: {e}")
            return

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"VIDEO_ERROR=could not open {args.video}")
            return

        frame_times = []
        n = 0
        t_start = time.perf_counter()
        while n < args.video_max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            model.predict(frame, device=device, verbose=False)
            frame_times.append((time.perf_counter() - t0) * 1000)
            n += 1
        t_total = time.perf_counter() - t_start
        cap.release()

        if n == 0:
            print("VIDEO_ERROR=zero frames read")
            return

        per_frame_fps = [1000.0 / t for t in frame_times if t > 0]
        print(f"VIDEO_FRAMES={n}")
        print(f"VIDEO_TOTAL_S={t_total:.2f}")
        print(f"VIDEO_AVG_MS={statistics.mean(frame_times):.2f}")
        print(f"VIDEO_AVG_FPS={n / t_total:.2f}")
        print(f"VIDEO_MIN_FPS={min(per_frame_fps):.2f}")
        print(f"VIDEO_MAX_FPS={max(per_frame_fps):.2f}")


if __name__ == "__main__":
    main()
