#!/usr/bin/env python3
"""
fairness.py — measurement methodology shared by the YOLO device suites.

Why this exists: the obvious way to compare backends gives wrong answers on the
kind of hardware these suites target. Three effects, all observed in practice on
an AMD Ryzen 7940HS while building this:

1. Thermal drift. Measuring backend A, then B, then C means C runs on a hotter
   chip. The same yolo11n ONNX-CPU inference measured 16.7 ms early in a session
   and 24.4 ms later — a 46% swing that has nothing to do with the backend.

2. Contention on shared-power SoCs. CPU and iGPU draw from one package power
   budget. Interleaving GPU and CPU measurements in one process made CPU
   inference 45% slower than measuring it alone. Whichever you do, be explicit:
   "alone" answers "which backend should I use?", "together" answers "what
   happens when both run?".

3. One-shot measurements. A single mean over 15 runs looked authoritative and
   was repeatable to ±40% across sessions.

So: measure in rounds, interleave within a round, take medians of medians, and
always report the spread. If the spread is wide, the number is not a number yet.

Use
---
    from fairness import Runner, measure, report

    runners = [
        Runner("cpu",        lambda: sess_cpu.run(...),  kind="cpu"),
        Runner("gpu eager",  gpu_eager,                  kind="gpu"),
        Runner("gpu fused",  gpu_compiled,               kind="gpu"),
    ]
    res = measure(runners, mode="isolated")   # or "interleaved"
    report(res)

`isolated` runs each backend to completion with a cooldown between them: the
right mode for "which backend is fastest". `interleaved` rotates between them
within each round: the right mode for cancelling drift when you must compare
backends of the same kind, and the honest mode for "what if they run together".
"""

import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Runner:
    """One thing to time. `fn` must be synchronous — if the device is async
    (CUDA/HIP/XPU streams), synchronise inside `fn` or the timing is fiction."""
    name: str
    fn: Callable[[], None]
    kind: str = "cpu"          # cpu | gpu | npu | other — used to warn about contention
    setup: Optional[Callable[[], None]] = None
    note: str = ""


@dataclass
class Result:
    name: str
    kind: str
    median_ms: float
    min_ms: float
    max_ms: float
    spread_pct: float          # (max-min)/median over the per-round medians
    rounds: List[float] = field(default_factory=list)
    note: str = ""
    stable: bool = True


def _time_reps(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def _warmup(runner, n):
    if runner.setup:
        runner.setup()
    for _ in range(n):
        runner.fn()


def measure(runners: List[Runner], mode: str = "isolated", rounds: int = 4,
            reps: int = 8, warmup: int = 8, cooldown_s: float = 2.0,
            stable_threshold_pct: float = 15.0, progress=None) -> List[Result]:
    """Time every runner. Returns one Result per runner.

    mode="isolated"    — each runner finishes before the next starts, with a
                         cooldown between. Use for "which backend is fastest".
    mode="interleaved" — rotate through runners inside each round, so thermal
                         drift and any background load hit all of them equally.
                         Use when comparing like with like, or to measure what
                         happens when backends genuinely run together.
    """
    if mode not in ("isolated", "interleaved"):
        raise ValueError("mode must be 'isolated' or 'interleaved'")
    say = progress or (lambda *_: None)
    per_round = {r.name: [] for r in runners}

    if mode == "interleaved":
        for r in runners:
            say(f"warming {r.name}")
            _warmup(r, warmup)
        for rnd in range(rounds):
            for r in runners:
                say(f"round {rnd + 1}/{rounds}: {r.name}")
                per_round[r.name].append(_time_reps(r.fn, reps))
    else:
        for r in runners:
            say(f"warming {r.name}")
            _warmup(r, warmup)
            for rnd in range(rounds):
                say(f"{r.name}: round {rnd + 1}/{rounds}")
                per_round[r.name].append(_time_reps(r.fn, reps))
            gc.collect()
            if cooldown_s:
                say(f"cooldown {cooldown_s}s after {r.name}")
                time.sleep(cooldown_s)

    out = []
    for r in runners:
        vals = per_round[r.name]
        med = statistics.median(vals)
        lo, hi = min(vals), max(vals)
        spread = 100.0 * (hi - lo) / med if med else 0.0
        out.append(Result(name=r.name, kind=r.kind, median_ms=med, min_ms=lo, max_ms=hi,
                          spread_pct=spread, rounds=vals, note=r.note,
                          stable=spread <= stable_threshold_pct))
    return out


def report(results: List[Result], baseline: Optional[str] = None, mode: str = "isolated",
           title: str = "") -> None:
    """Print a table. `baseline` names the runner everything is compared against;
    default is the slowest, so speedups read as ">1 is better"."""
    if not results:
        print("no results")
        return
    base = next((r for r in results if r.name == baseline), None) or max(results, key=lambda r: r.median_ms)
    width = max(len(r.name) for r in results) + 2

    if title:
        print(f"\n{title}")
    print(f"{'backend':{width}} {'median ms':>10} {'min':>8} {'max':>8} {'spread':>8} {'vs ' + base.name:>14}")
    print("-" * (width + 52))
    for r in sorted(results, key=lambda r: r.median_ms):
        flag = "" if r.stable else "  <- unstable, re-run"
        print(f"{r.name:{width}} {r.median_ms:10.2f} {r.min_ms:8.2f} {r.max_ms:8.2f} "
              f"{r.spread_pct:7.1f}% {base.median_ms / r.median_ms:13.2f}x{flag}")

    unstable = [r.name for r in results if not r.stable]
    if unstable:
        print(f"\n  Unstable (>15% spread across rounds): {', '.join(unstable)}.")
        print("  Treat these as indicative only. Usual causes: thermal throttling, another")
        print("  workload on the machine, or a first-round compile that warmup did not absorb.")

    kinds = {r.kind for r in results}
    if mode == "interleaved" and len(kinds & {"cpu", "gpu", "npu"}) > 1:
        print("\n  Note: interleaved mode measured CPU and accelerator backends against each")
        print("  other in one process. On an SoC they share a package power budget, so these")
        print("  numbers describe running them TOGETHER, not each at its best. For 'which")
        print("  backend should I use', re-run with mode='isolated'.")


def csv_rows(results: List[Result], model: str, stage: str = "fair_compare"):
    """Rows for the suite's benchmark_results.csv: backend,model,stage,metric,value"""
    rows = []
    for r in results:
        rows.append(f"{r.name.replace(' ', '_')},{model},{stage},median_ms,{r.median_ms:.2f}")
        rows.append(f"{r.name.replace(' ', '_')},{model},{stage},spread_pct,{r.spread_pct:.1f}")
    return rows


if __name__ == "__main__":
    # Self-test with synthetic load, so the module can be sanity-checked anywhere.
    import math

    def busy(n):
        return lambda: sum(math.sqrt(i) for i in range(n))

    rs = [Runner("fast", busy(20000), kind="cpu"), Runner("slow", busy(60000), kind="cpu")]
    report(measure(rs, mode="isolated", rounds=3, reps=4, warmup=2, cooldown_s=0),
           mode="isolated", title="fairness.py self-test (synthetic load)")
