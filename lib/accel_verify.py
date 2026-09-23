#!/usr/bin/env python3
"""
accel_verify.py — did the accelerator actually do the work?

A backend that reports "GPU" or "NPU" is claiming, not proving. Every runtime in
this space will silently fall back to the CPU: a wrong package shadowing the
accelerated one, a driver that failed to load, an unsupported op, a permissions
problem on the device node. The benchmark still produces plausible numbers, and
they are CPU numbers.

This reads the *kernel's* own per-device accounting around a workload and tells
you whether the silicon moved. It is the ground truth the suites check against.

Usage
-----
    # watch a process that is already running
    python3 accel_verify.py --pid 12345 --seconds 5

    # run a command and watch it
    python3 accel_verify.py --seconds 5 -- python infer.py --device gpu

    # just show what this machine exposes
    python3 accel_verify.py --list

    # machine-readable, for a test suite
    python3 accel_verify.py --pid 12345 --seconds 5 --json

Exit status is 0 if at least one accelerator engine was measurably busy, 1 if
none were (i.e. the work almost certainly ran on the CPU), 2 on usage errors.

Counter sources, by platform
----------------------------
AMD (amdgpu)        /proc/<pid>/fdinfo/*  drm-engine-{gfx,compute,enc,dec}
                    per-process, exact. Also drm-memory-vram.
Intel (i915 / xe)   /proc/<pid>/fdinfo/*  drm-engine-{render,compute,rcs,ccs,...}
                    per-process, exact.
Intel NPU (ivpu)    /sys/**/npu_busy_time_us — device-wide, not per process.
Rockchip NPU        /sys/kernel/debug/rknpu/load — device-wide ("Core0: 12%,").
Mali GPU            /sys/class/devfreq/*.gpu/load — device-wide ("42@600000000Hz").
Qualcomm Adreno     /sys/class/kgsl/kgsl-3d0/gpubusy — device-wide (busy total).
Qualcomm Hexagon    no kernel counter is exposed; see note in report().

Device-wide counters cannot separate your process from anything else using the
device, so they are reported as such. Per-process counters are exact.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# per-process counters (exact): DRM fdinfo, used by amdgpu / i915 / xe
# --------------------------------------------------------------------------

_ENGINE_RE = re.compile(r"^drm-engine-([a-z0-9_-]+):\s+(\d+)\s*ns", re.I | re.M)
_MEM_RE = re.compile(r"^drm-(?:memory|resident)-(vram|local0|gtt|system):\s+(\d+)\s*([KMG]i?B)", re.I | re.M)
_CLIENT_RE = re.compile(r"^drm-client-id:\s*(\d+)", re.M)
_DRIVER_RE = re.compile(r"^drm-driver:\s*(\S+)", re.M)

_UNIT = {"kib": 1024, "mib": 1024**2, "gib": 1024**3, "kb": 1000, "mb": 1000**2, "gb": 1000**3}

# Engines that mean "this did real compute/graphics work", as opposed to the
# video codec block, which is a separate unit and separately interesting.
_COMPUTE_ENGINES = ("gfx", "compute", "render", "rcs", "ccs", "cs")
_VIDEO_ENGINES = ("enc", "dec", "video", "vcn", "vcs", "vecs", "video-enhance", "bsd")


def _pids_of(pid, include_children=True):
    pids = [pid]
    if include_children:
        try:
            out = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)],
                                 capture_output=True, text=True, timeout=5).stdout
            pids += [int(p) for p in out.split()]
        except Exception:
            pass
    return pids


def read_drm_fdinfo(pid):
    """Sum engine-busy ns and memory per DRM client for one process.

    A process can hold several fds onto the same DRM client; drm-client-id
    dedupes them, otherwise the same work is counted twice.
    """
    result = {"driver": None, "engines": {}, "memory": {}}
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return None
    seen_clients = set()
    found = False
    for fd in fds:
        try:
            if not os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/dri/"):
                continue
            text = open(f"/proc/{pid}/fdinfo/{fd}").read()
        except OSError:
            continue
        drv = _DRIVER_RE.search(text)
        if not drv:
            continue
        cid = _CLIENT_RE.search(text)
        key = cid.group(1) if cid else f"fd{fd}"
        if key in seen_clients:
            continue
        seen_clients.add(key)
        found = True
        result["driver"] = drv.group(1)
        for name, ns in _ENGINE_RE.findall(text):
            result["engines"][name.lower()] = result["engines"].get(name.lower(), 0) + int(ns)
        for region, val, unit in _MEM_RE.findall(text):
            result["memory"][region.lower()] = result["memory"].get(region.lower(), 0) + int(val) * _UNIT.get(unit.lower(), 1)
    return result if found else None


# --------------------------------------------------------------------------
# device-wide counters: Intel NPU, Rockchip NPU, Mali, Adreno
# --------------------------------------------------------------------------

def _read_first(patterns):
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            try:
                return path, open(path).read().strip()
            except OSError:
                continue
    return None, None


def read_device_counters():
    """Device-wide accelerator counters. Values are raw; deltas are taken later."""
    out = {}

    # Intel NPU (intel_vpu): cumulative busy microseconds
    path, val = _read_first([
        "/sys/devices/pci0000:00/*/accel/accel*/device/npu_busy_time_us",
        "/sys/devices/pci0000:00/*/npu_busy_time_us",
        "/sys/class/accel/accel*/device/npu_busy_time_us",
        "/sys/bus/pci/drivers/intel_vpu/*/npu_busy_time_us",
    ])
    if val is not None:
        try:
            out["intel_npu"] = {"kind": "cumulative_us", "value": int(val), "path": path, "scope": "device"}
        except ValueError:
            pass

    # Rockchip RKNPU: instantaneous per-core percentage
    path, val = _read_first(["/sys/kernel/debug/rknpu/load", "/sys/class/devfreq/*.npu/load"])
    if val is not None:
        pcts = [int(p) for p in re.findall(r"(\d+)%", val)]
        if not pcts:
            m = re.match(r"^(\d+)@", val)
            pcts = [int(m.group(1))] if m else []
        if pcts:
            out["rockchip_npu"] = {"kind": "instant_pct", "value": max(pcts), "cores": pcts, "path": path, "scope": "device"}

    # Mali GPU (Rockchip and others): "42@600000000Hz"
    path, val = _read_first(["/sys/class/devfreq/*.gpu/load", "/sys/class/devfreq/*gpu*/load"])
    if val is not None:
        m = re.match(r"^(\d+)", val)
        if m:
            out["mali_gpu"] = {"kind": "instant_pct", "value": int(m.group(1)), "path": path, "scope": "device"}

    # Qualcomm Adreno: "<busy> <total>" in microseconds since last read (destructive read)
    path, val = _read_first(["/sys/class/kgsl/kgsl-3d0/gpubusy"])
    if val is not None:
        parts = val.split()
        if len(parts) >= 2:
            try:
                busy, total = int(parts[0]), int(parts[1])
                out["adreno_gpu"] = {"kind": "ratio", "busy": busy, "total": total,
                                     "value": (100.0 * busy / total) if total else 0.0,
                                     "path": path, "scope": "device",
                                     "note": "reading resets the counter"}
            except ValueError:
                pass
    if "adreno_gpu" not in out:
        path, val = _read_first(["/sys/class/kgsl/kgsl-3d0/devfreq/gpu_load"])
        if val is not None and val.isdigit():
            out["adreno_gpu"] = {"kind": "instant_pct", "value": int(val), "path": path, "scope": "device"}

    # AMD 3D engine (not the media block); useful as a coarse cross-check
    path, val = _read_first(["/sys/class/drm/card*/device/gpu_busy_percent"])
    if val is not None and val.isdigit():
        out["amdgpu_3d"] = {"kind": "instant_pct", "value": int(val), "path": path, "scope": "device"}

    return out


def hexagon_available():
    """Qualcomm Hexagon/HTP: is the DSP reachable at all?

    There is no kernel-exposed busy counter for the HTP. Presence of the fastrpc
    device nodes tells you the path exists; actual utilisation has to come from
    QNN's own profiling (QNN_LOG_LEVEL / profiling_level=detailed).
    """
    nodes = sorted(glob.glob("/dev/fastrpc-*") + glob.glob("/dev/adsprpc-smd") + glob.glob("/dev/subsys_*"))
    return nodes


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def sample(pids):
    snap = {"t": time.time(), "per_process": {}, "device": read_device_counters()}
    for pid in pids:
        fi = read_drm_fdinfo(pid)
        if fi:
            snap["per_process"][pid] = fi
    return snap


def diff(a, b):
    """Turn two snapshots into utilisation."""
    wall_ns = max(1e-9, (b["t"] - a["t"])) * 1e9
    out = {"seconds": round(b["t"] - a["t"], 2), "per_process": {}, "device": {}, "busy": False}

    for pid, cur in b["per_process"].items():
        prev = a["per_process"].get(pid)
        if not prev:
            continue
        engines = {}
        for name, ns in cur["engines"].items():
            d = ns - prev["engines"].get(name, 0)
            if d > 0:
                engines[name] = round(100.0 * d / wall_ns, 1)
        if engines or cur["memory"]:
            out["per_process"][pid] = {
                "driver": cur["driver"],
                "engine_percent": engines,
                "compute_percent": round(sum(v for k, v in engines.items()
                                             if any(k.startswith(e) for e in _COMPUTE_ENGINES)), 1),
                "video_percent": round(sum(v for k, v in engines.items()
                                           if any(k.startswith(e) for e in _VIDEO_ENGINES)), 1),
                "memory": cur["memory"],
            }
            if engines:
                out["busy"] = True

    for key, cur in b["device"].items():
        prev = a["device"].get(key)
        if cur["kind"] == "cumulative_us" and prev:
            d = cur["value"] - prev["value"]
            pct = round(100.0 * (d * 1000.0) / wall_ns, 1)
            out["device"][key] = {"percent": pct, "scope": "device", "path": cur["path"]}
            if pct > 1.0:
                out["busy"] = True
        elif cur["kind"] == "ratio":
            pct = round(cur["value"], 1)
            out["device"][key] = {"percent": pct, "scope": "device", "path": cur["path"]}
            if pct > 1.0:
                out["busy"] = True
        elif cur["kind"] == "instant_pct":
            # instantaneous: report the higher of the two samples
            pct = max(cur["value"], prev["value"] if prev else 0)
            out["device"][key] = {"percent": pct, "scope": "device", "path": cur["path"], "kind": "instantaneous"}
            if pct > 1.0 and key != "amdgpu_3d":
                out["busy"] = True
    return out


def report(res, json_out=False):
    if json_out:
        print(json.dumps(res, indent=2))
        return
    print(f"measured over {res['seconds']} s")
    if res["per_process"]:
        print("\nper-process (exact, from the kernel's DRM accounting):")
        for pid, p in res["per_process"].items():
            eng = ", ".join(f"{k} {v}%" for k, v in sorted(p["engine_percent"].items())) or "idle"
            mem = ", ".join(f"{k} {v // (1024*1024)} MB" for k, v in p["memory"].items())
            print(f"  pid {pid} [{p['driver']}]  {eng}")
            if mem:
                print(f"        memory: {mem}")
    else:
        print("\nper-process: no DRM clients found for these pids")
        print("  (normal for NPU-only or OpenCL-only backends, which do not appear in DRM fdinfo)")

    if res["device"]:
        print("\ndevice-wide (cannot separate your process from other users of the device):")
        for k, v in sorted(res["device"].items()):
            note = " [instantaneous sample]" if v.get("kind") == "instantaneous" else ""
            print(f"  {k:14} {v['percent']:6.1f}%{note}   {v['path']}")

    hexes = hexagon_available()
    if hexes:
        print(f"\nQualcomm fastrpc nodes present: {', '.join(hexes)}")
        print("  The kernel exposes no HTP busy counter. To prove the NPU ran, use QNN's own")
        print("  profiling (profiling_level='detailed') or compare HTP vs CPU backend latency.")

    print()
    if res["busy"]:
        print("VERDICT: an accelerator engine was measurably busy — the work did NOT run only on the CPU.")
    else:
        print("VERDICT: no accelerator engine showed activity.")
        print("  Either the workload really ran on the CPU, or this platform exposes no counter")
        print("  for the unit in use (Hexagon HTP, some OpenCL stacks). Treat a 'GPU'/'NPU' claim")
        print("  from the runtime as unproven until you can show a counter moving.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, help="process to watch (defaults to the command given after --)")
    ap.add_argument("--seconds", type=float, default=5.0, help="measurement window (default 5)")
    ap.add_argument("--list", action="store_true", help="show the counters this machine exposes, then exit")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-children", action="store_true", help="do not include child processes")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run and watch")
    args = ap.parse_args()

    if args.list:
        dev = read_device_counters()
        print("device-wide counters found:" if dev else "no device-wide accelerator counters found")
        for k, v in sorted(dev.items()):
            print(f"  {k:14} {v['path']}")
        hexes = hexagon_available()
        if hexes:
            print(f"fastrpc (Qualcomm DSP) nodes: {', '.join(hexes)}")
        print("\nper-process DRM accounting is available for any process holding /dev/dri/* "
              "(amdgpu, i915, xe).")
        return 0

    proc = None
    cmd = [c for c in args.cmd if c != "--"]
    if cmd:
        proc = subprocess.Popen(cmd)
        pid = proc.pid
    elif args.pid:
        pid = args.pid
    else:
        ap.error("give --pid, or a command after --")
        return 2

    pids = _pids_of(pid, include_children=not args.no_children)
    a = sample(pids)
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        if proc and proc.poll() is not None:
            break
        time.sleep(min(0.25, max(0.0, deadline - time.time())))
    pids = _pids_of(pid, include_children=not args.no_children)  # children may have appeared
    b = sample(pids)

    if proc:
        try:
            proc.wait(timeout=max(1.0, args.seconds))
        except subprocess.TimeoutExpired:
            proc.terminate()

    res = diff(a, b)
    report(res, json_out=args.json)
    return 0 if res["busy"] else 1


if __name__ == "__main__":
    sys.exit(main())
