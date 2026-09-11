"""Runtime and memory instrumentation for pipeline stages.

Usage:
    prof = Profiler("ingest")
    with prof.stage("extraction", rows=n):
        ...
    prof.save(project)      # appends to .powermet/cache/profiles.jsonl
    print(prof.render())

Peak memory is the process peak RSS from resource.getrusage (monotonic over the process
lifetime), sampled at the end of each stage; Python-heap peak comes from tracemalloc when
enabled (slower, off by default).
"""

from __future__ import annotations

import json
import os
import platform
import resource
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from powermet.textfmt import table


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return ru / (1024 * 1024) if platform.system() == "Darwin" else ru / 1024


def current_rss_mb() -> float:
    try:
        import psutil  # optional

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return peak_rss_mb()


@dataclass
class StageRecord:
    stage: str
    wall_s: float
    cpu_s: float
    peak_rss_mb: float
    heap_peak_mb: float | None = None
    rows: int | None = None
    detail: str = ""


@dataclass
class Profiler:
    command: str
    stages: list[StageRecord] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    trace_heap: bool = False
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    @contextmanager
    def stage(self, name: str, rows: int | None = None, detail: str = ""):
        if self.trace_heap:
            tracemalloc.start()
        t0, c0 = time.perf_counter(), time.process_time()
        rec = StageRecord(name, 0.0, 0.0, 0.0, rows=rows, detail=detail)
        self.stages.append(rec)
        try:
            yield rec
        finally:
            rec.wall_s = time.perf_counter() - t0
            rec.cpu_s = time.process_time() - c0
            rec.peak_rss_mb = peak_rss_mb()
            if self.trace_heap:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rec.heap_peak_mb = peak / (1024 * 1024)

    @property
    def total_wall_s(self) -> float:
        return time.perf_counter() - self._t0

    def to_dict(self) -> dict:
        return {
            "command": self.command, "started_at": self.started_at, "python": sys.version.split()[0],
            "platform": platform.platform(), "total_wall_s": self.total_wall_s,
            "peak_rss_mb": peak_rss_mb(), "stages": [asdict(s) for s in self.stages],
        }

    def save(self, project) -> Path:
        from powermet.catalog import db_path, record_profile

        record_profile(project, self.to_dict())
        return db_path(project)

    def render(self) -> str:
        return render_profile(self.to_dict())


def aggregate_stages(stages: list[dict]) -> list[dict]:
    """Sum repeated stages (one per run directory) into one line per stage name, in first-seen order."""
    agg: dict[str, dict] = {}
    for s in stages:
        a = agg.setdefault(s["stage"], {"stage": s["stage"], "wall_s": 0.0, "cpu_s": 0.0, "peak_rss_mb": 0.0, "rows": 0, "calls": 0})
        a["wall_s"] += s["wall_s"]
        a["cpu_s"] += s["cpu_s"]
        a["peak_rss_mb"] = max(a["peak_rss_mb"], s["peak_rss_mb"])
        a["rows"] += s.get("rows") or 0
        a["calls"] += 1
    return list(agg.values())


def render_profile(d: dict) -> str:
    rows = []
    for s in aggregate_stages(d["stages"]):
        rows.append([s["stage"], f"{s['wall_s']:.2f} s", f"{s['cpu_s']:.2f} s", f"{s['peak_rss_mb']:.0f} MB",
                     f"{s['rows']:,}" if s["rows"] else "", str(s["calls"]) if s["calls"] > 1 else ""])
    out = [f"Runtime profile: {d['command']}  ({d['started_at']})", "",
           table(["Stage", "Wall", "CPU", "Peak RSS", "Rows", "Calls"], rows, ["l", "r", "r", "r", "r", "r"]), "",
           f"Total wall time: {d['total_wall_s']:.2f} s    Peak memory (RSS): {d['peak_rss_mb']:.0f} MB    Python {d['python']}"]
    return "\n".join(out)


def load_profiles(project, last: int = 5) -> list[dict]:
    from powermet.catalog import profiles

    return profiles(project, last=last)
