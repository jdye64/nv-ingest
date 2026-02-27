from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, Optional

import pandas as pd


def estimate_batch_rows(batch: Any) -> Optional[int]:
    try:
        if isinstance(batch, pd.DataFrame):
            return int(len(batch.index))
        if isinstance(batch, (list, tuple)):
            return int(len(batch))
    except Exception:
        return None
    return None


def emit_actor_metrics(
    metrics_actor: Any,
    *,
    stage: str,
    duration_sec: float,
    input_rows: Optional[int],
    output_rows: Optional[int],
    ok: bool,
    error: BaseException | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if metrics_actor is None:
        return
    try:
        metrics_actor.record.remote(
            stage=str(stage),
            duration_sec=float(duration_sec),
            input_rows=input_rows,
            output_rows=output_rows,
            ok=bool(ok),
            error_type=(error.__class__.__name__ if error is not None else None),
            error_message=(str(error) if error is not None else None),
            metadata=dict(metadata or {}),
        )
    except Exception:
        # Metrics are best-effort and must never break ingestion.
        pass


class MetricsActor:
    def __init__(self, run_id: str = "unknown") -> None:
        self.reset(run_id=run_id)

    def reset(self, run_id: str = "unknown") -> None:
        self.run_id = str(run_id or "unknown")
        self.started_at_epoch_s = float(time.time())
        self.total_calls = 0
        self.total_errors = 0
        self.total_duration_sec = 0.0
        self.total_input_rows = 0
        self.total_output_rows = 0
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.error_types: Dict[str, int] = defaultdict(int)

    def record(
        self,
        *,
        stage: str,
        duration_sec: float,
        input_rows: Optional[int] = None,
        output_rows: Optional[int] = None,
        ok: bool = True,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        _ = (error_message, metadata)
        stage_key = str(stage or "unknown_stage")
        bucket = self.stages.setdefault(
            stage_key,
            {
                "calls": 0,
                "errors": 0,
                "duration_sec": 0.0,
                "min_duration_sec": None,
                "max_duration_sec": None,
                "input_rows": 0,
                "output_rows": 0,
                "error_types": defaultdict(int),
            },
        )

        d = float(duration_sec)
        bucket["calls"] += 1
        bucket["duration_sec"] += d
        bucket["min_duration_sec"] = d if bucket["min_duration_sec"] is None else min(bucket["min_duration_sec"], d)
        bucket["max_duration_sec"] = d if bucket["max_duration_sec"] is None else max(bucket["max_duration_sec"], d)

        in_rows = int(input_rows) if input_rows is not None else 0
        out_rows = int(output_rows) if output_rows is not None else 0
        bucket["input_rows"] += in_rows
        bucket["output_rows"] += out_rows

        self.total_calls += 1
        self.total_duration_sec += d
        self.total_input_rows += in_rows
        self.total_output_rows += out_rows

        if not ok:
            bucket["errors"] += 1
            self.total_errors += 1
            if error_type:
                et = str(error_type)
                bucket["error_types"][et] += 1
                self.error_types[et] += 1

    def report(self) -> Dict[str, Any]:
        finished = float(time.time())
        stage_report: Dict[str, Dict[str, Any]] = {}
        for stage, data in self.stages.items():
            calls = int(data.get("calls", 0))
            duration = float(data.get("duration_sec", 0.0))
            stage_report[stage] = {
                "calls": calls,
                "errors": int(data.get("errors", 0)),
                "duration_sec": duration,
                "avg_duration_sec": (duration / calls) if calls > 0 else 0.0,
                "min_duration_sec": data.get("min_duration_sec"),
                "max_duration_sec": data.get("max_duration_sec"),
                "input_rows": int(data.get("input_rows", 0)),
                "output_rows": int(data.get("output_rows", 0)),
                "error_types": dict(data.get("error_types", {})),
            }

        total_calls = int(self.total_calls)
        total_duration = float(self.total_duration_sec)
        return {
            "run_id": self.run_id,
            "started_at_epoch_s": self.started_at_epoch_s,
            "finished_at_epoch_s": finished,
            "wall_time_sec": float(max(0.0, finished - self.started_at_epoch_s)),
            "totals": {
                "calls": total_calls,
                "errors": int(self.total_errors),
                "duration_sec": total_duration,
                "avg_duration_sec": (total_duration / total_calls) if total_calls > 0 else 0.0,
                "input_rows": int(self.total_input_rows),
                "output_rows": int(self.total_output_rows),
                "error_types": dict(self.error_types),
            },
            "stages": stage_report,
        }
