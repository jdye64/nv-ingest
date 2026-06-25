#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a reproducible local-GPU performance suite for DGX Spark experiments.

The suite is intentionally a thin orchestrator over the repository CLIs:

* ``retriever benchmark all run`` for actor-stage throughput.
* ``retriever harness run`` for end-to-end ingest/embed/LanceDB latency.
* ``retriever query`` as a post-ingest smoke check when an E2E run succeeds.

Outputs are written under ``nemo_retriever/artifacts`` by default, which is
already ignored by this repository.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/home/jdyer/datasets/bo20")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "nemo_retriever" / "artifacts" / "dgx_spark_perf"
DEFAULT_PRESET = "PE_GE_OCR_TE_DENSE"
DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_QUERY = "What are the main topics covered in these documents?"
VDB_DROP_RE = re.compile(
    r"accepted=(?P<accepted>\d+)\s+"
    r"dropped_no_embedding=(?P<dropped_no_embedding>\d+)\s+"
    r"dropped_bad_length=(?P<dropped_bad_length>\d+)\s+"
    r"dropped_no_text=(?P<dropped_no_text>\d+)"
)
PREPARED_VDB_RE = re.compile(r"Prepared (?P<prepared>\d+) uploadable VDB records")


@dataclass
class CommandResult:
    name: str
    command: list[str]
    return_code: int
    elapsed_seconds: float
    log_path: str

    @property
    def success(self) -> bool:
        return self.return_code == 0


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")


def short_command(command: Sequence[str]) -> str:
    return " ".join(command)


def run_command(
    *,
    name: str,
    command: Sequence[str],
    log_path: Path,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if dry_run:
        log_path.write_text("[dry-run] " + short_command(command) + "\n", encoding="utf-8")
        return CommandResult(name, list(command), 0, 0.0, str(log_path))

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("$ " + short_command(command) + "\n\n")
        proc = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = proc.wait()

    elapsed = time.perf_counter() - start
    return CommandResult(name, list(command), int(return_code), float(elapsed), str(log_path))


def capture_command(command: Sequence[str], *, cwd: Path = REPO_ROOT) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return {"command": list(command), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": list(command),
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_environment() -> dict[str, Any]:
    return {
        "timestamp_utc": utc_stamp(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "sys_executable": sys.executable,
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "python": sys.version.replace("\n", " "),
        "git_head": capture_command(["git", "rev-parse", "HEAD"]),
        "git_branch": capture_command(["git", "branch", "--show-current"]),
        "git_status": capture_command(["git", "status", "--short", "--branch"]),
        "nvidia_smi": capture_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
        "lscpu": capture_command(["lscpu"]),
        "uv_version": capture_command(["uv", "--version"]),
    }


def active_venv_dir() -> Path | None:
    """Return the active virtualenv directory, preserving symlinked venv Python paths."""
    candidates: list[Path] = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates.append(Path(sys.prefix))
    executable = Path(sys.executable)
    if executable.parent.name == "bin":
        candidates.append(executable.parent.parent)

    for candidate in candidates:
        if (candidate / "pyvenv.cfg").exists():
            return candidate
    return None


def current_venv_env() -> dict[str, str]:
    """Return env vars that make child processes behave like this venv is active."""
    venv_dir = active_venv_dir()
    if venv_dir is None:
        return {}
    bin_dir = venv_dir / "bin"
    return {
        "VIRTUAL_ENV": str(venv_dir),
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
    }


def retriever_cli() -> str:
    """Resolve the retriever console script from the active Python environment."""
    suffix = ".exe" if os.name == "nt" else ""
    venv_dir = active_venv_dir()
    candidates = []
    if venv_dir is not None:
        candidates.append(venv_dir / "bin" / f"retriever{suffix}")
    candidates.append(Path(sys.executable).parent / f"retriever{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which(f"retriever{suffix}")
    if resolved:
        return resolved

    raise FileNotFoundError(
        "Could not find the retriever console script. Run this benchmark with the repo .venv Python interpreter."
    )


def python_main_command(module: str) -> list[str]:
    """Build a command that invokes a Typer main() without the root CLI imports."""
    return [sys.executable, "-c", f"from {module} import main; main()"]


def top_level_pdfs(dataset: Path) -> list[Path]:
    dataset = dataset.expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset}")
    return sorted(p for p in dataset.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def create_input_manifest(dataset: Path, run_dir: Path) -> tuple[Path, list[Path]]:
    pdfs = top_level_pdfs(dataset)
    if not pdfs:
        raise FileNotFoundError(f"No top-level PDFs found in {dataset}")

    input_dir = run_dir / "inputs" / dataset.name
    input_dir.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        target = input_dir / pdf.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(pdf)
    return input_dir, pdfs


def choose_microbench_pdf(pdfs: Sequence[Path]) -> Path:
    if not pdfs:
        raise ValueError("No PDFs available for microbenchmark selection")
    return max(pdfs, key=lambda p: p.stat().st_size)


def write_harness_config(
    *,
    path: Path,
    dataset_dir: Path,
    harness_artifacts_dir: Path,
    preset: str,
    embed_model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep this YAML explicit. The custom dataset must not inherit jp20's BEIR
    # query CSV from the repository default active config.
    path.write_text(
        f"""# Generated by scripts/dgx_spark_perf_benchmark.py
active:
  dataset: bo20_local
  preset: {preset}
  run_mode: inprocess
  query_csv: null
  input_type: pdf
  recall_required: false
  evaluation_mode: none
  beir_loader: null
  beir_dataset_name: null
  beir_split: test
  beir_query_language: null
  beir_doc_id_field: pdf_page
  beir_ks: [1, 3, 5, 10]
  artifacts_dir: {harness_artifacts_dir}
  ray_address: null
  lancedb_uri: lancedb
  hybrid: false
  embed_model_name: {embed_model_name}
  embed_modality: text
  embed_granularity: element
  extract_page_as_image: false
  extract_infographics: false
  write_detection_file: false
  use_heuristics: false

presets:
  {preset}:
    pdf_extract_workers: 8
    pdf_extract_num_cpus: 2.0
    pdf_extract_batch_size: 4
    pdf_split_batch_size: 1
    page_elements_batch_size: 4
    page_elements_workers: 3
    ocr_workers: 1
    ocr_batch_size: 8
    embed_workers: 1
    embed_batch_size: 128
    embed_enforce_eager: true
    embed_max_length: 2048
    page_elements_cpus_per_actor: 1.0
    ocr_cpus_per_actor: 1.0
    embed_cpus_per_actor: 1.0
    gpu_page_elements: 0.1
    gpu_ocr: 0.1
    gpu_embed: 0.25
    embed_modality: text
    embed_model_name: {embed_model_name}

datasets:
  bo20_local:
    path: {dataset_dir}
    query_csv: null
    input_type: pdf
    recall_required: false
    evaluation_mode: none
""",
        encoding="utf-8",
    )


def newest_harness_result(harness_artifacts_dir: Path, run_name: str) -> dict[str, Any] | None:
    matches = sorted(harness_artifacts_dir.glob(f"{run_name}_*/results.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def load_stage_results(stage_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in sorted(stage_dir.glob("*.json")):
        try:
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            out[path.stem] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def summarize_numeric(values: Iterable[float]) -> dict[str, float] | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    p50 = statistics.median(vals)
    if len(vals) == 1:
        p95 = vals[0]
    else:
        p95 = vals[int(round((len(vals) - 1) * 0.95))]
    return {
        "min": vals[0],
        "p50": p50,
        "p95": p95,
        "max": vals[-1],
    }


def _count_lancedb_rows(artifact_dir: Path) -> int | None:
    try:
        import lancedb  # type: ignore[import-not-found]

        db = lancedb.connect(str(artifact_dir / "lancedb"))
        table = db.open_table("nv-ingest")
        return int(table.count_rows())
    except Exception:
        return None


def summarize_embedding_log_quality(
    results: Sequence[CommandResult], e2e_metrics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    totals = {
        "accepted": 0,
        "dropped_no_embedding": 0,
        "dropped_bad_length": 0,
        "dropped_no_text": 0,
    }
    validation_error_count = 0
    matched_drop_lines = 0
    prepared_records = 0
    scanned_logs: list[str] = []

    for result in results:
        path = Path(result.log_path)
        if not path.exists():
            continue
        scanned_logs.append(str(path))
        text = path.read_text(encoding="utf-8", errors="replace")
        validation_error_count += text.count("VLLMValidationError")
        prepared_records += sum(int(match.group("prepared")) for match in PREPARED_VDB_RE.finditer(text))
        for match in VDB_DROP_RE.finditer(text):
            matched_drop_lines += 1
            for key in totals:
                totals[key] += int(match.group(key))

    persisted_records = 0
    counted_tables = 0
    for metrics in e2e_metrics.values():
        artifact_dir = metrics.get("artifact_dir")
        if not artifact_dir:
            continue
        count = _count_lancedb_rows(Path(str(artifact_dir)))
        if count is None:
            continue
        persisted_records += count
        counted_tables += 1

    dropped = totals["dropped_no_embedding"] + totals["dropped_bad_length"]
    status = "passed"
    if dropped or validation_error_count:
        status = "failed"
    elif prepared_records and counted_tables and persisted_records < prepared_records:
        status = "failed"
    elif not matched_drop_lines and not (prepared_records and counted_tables):
        status = "skipped"

    detail = (
        f"accepted={totals['accepted']} dropped_no_embedding={totals['dropped_no_embedding']} "
        f"dropped_bad_length={totals['dropped_bad_length']} dropped_no_text={totals['dropped_no_text']} "
        f"vllm_validation_errors={validation_error_count} prepared_records={prepared_records} "
        f"persisted_records={persisted_records} scanned_logs={len(scanned_logs)}"
    )
    return {
        "name": "embedding_log_quality",
        "status": status,
        "detail": detail,
        "metrics": {
            **totals,
            "vllm_validation_errors": validation_error_count,
            "matched_drop_lines": matched_drop_lines,
            "prepared_records": prepared_records,
            "persisted_records": persisted_records,
            "counted_lancedb_tables": counted_tables,
            "scanned_logs": scanned_logs,
        },
    }


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# DGX Spark Performance Benchmark",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- Input manifest: `{summary['input_manifest']}`",
        f"- PDF count: `{summary['pdf_count']}`",
        f"- Microbenchmark PDF: `{summary['microbench_pdf']}`",
        "",
        "## Commands",
        "",
    ]
    for result in summary["commands"]:
        status = "PASS" if result["return_code"] == 0 else "FAIL"
        lines.append(
            f"- `{result['name']}`: {status}, {result['elapsed_seconds']:.2f}s, log `{result['log_path']}`"
        )

    lines.extend(["", "## E2E Metrics", ""])
    for name, metrics in summary.get("e2e_metrics", {}).items():
        lines.append(f"- `{name}`: `{json.dumps(metrics, sort_keys=True)}`")

    if summary.get("latency_summary"):
        lines.extend(["", "## Latency Summary", ""])
        for key, metrics in summary["latency_summary"].items():
            lines.append(f"- `{key}`: `{json.dumps(metrics, sort_keys=True)}`")

    lines.extend(["", "## Quality Checks", ""])
    for check in summary.get("quality_checks", []):
        lines.append(f"- `{check['name']}`: {check['status']} - {check['detail']}")

    if summary.get("validation_errors"):
        lines.extend(["", "## Validation Errors", ""])
        for error in summary["validation_errors"]:
            lines.append(f"- {error}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_e2e_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for name, metrics in summary.get("e2e_metrics", {}).items():
        rows.append(
            {
                "name": name,
                "success": metrics.get("success"),
                "ingest_secs": metrics.get("ingest_secs"),
                "pages": metrics.get("pages"),
                "pages_per_sec_ingest": metrics.get("pages_per_sec_ingest"),
                "artifact_dir": metrics.get("artifact_dir"),
            }
        )
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Directory containing top-level PDFs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Benchmark artifact root.")
    parser.add_argument("--run-id", default=None, help="Stable run ID. Defaults to a UTC timestamp.")
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="Harness preset name to generate.")
    parser.add_argument("--embed-model-name", default=DEFAULT_EMBED_MODEL, help="Local embedding model name.")
    parser.add_argument("--stage-repeats", type=int, default=1, help="Number of all-stage microbenchmark repeats.")
    parser.add_argument("--e2e-repeats", type=int, default=1, help="Number of fixed-preset E2E repeats.")
    parser.add_argument(
        "--heuristic-e2e-repeats",
        type=int,
        default=1,
        help="Number of E2E repeats using harness/pipeline resource heuristics.",
    )
    parser.add_argument("--stage-workers", default="1,2", help="Worker counts for stage microbenchmarks.")
    parser.add_argument("--stage-batch-sizes", default="1,2,4,8", help="Batch sizes for stage microbenchmarks.")
    parser.add_argument("--rows-split", type=int, default=256)
    parser.add_argument("--rows-extract", type=int, default=256)
    parser.add_argument("--rows-page-elements", type=int, default=256)
    parser.add_argument("--rows-ocr", type=int, default=128)
    parser.add_argument("--stage-num-gpus", type=float, default=0.5)
    parser.add_argument("--stage-num-cpus", type=float, default=1.0)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Post-ingest smoke query.")
    parser.add_argument("--skip-stage", action="store_true", help="Skip actor-stage microbenchmarks.")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip E2E harness runs.")
    parser.add_argument("--skip-query-smoke", action="store_true", help="Skip post-ingest query smoke checks.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands without executing them.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage_repeats < 0 or args.e2e_repeats < 0 or args.heuristic_e2e_repeats < 0:
        raise ValueError("Repeat counts must be >= 0")

    run_id = args.run_id or f"dgx_spark_bo20_{utc_stamp()}"
    run_dir = args.output_root.expanduser().resolve() / run_id
    logs_dir = run_dir / "logs"
    stage_root = run_dir / "stage_microbench"
    harness_artifacts_dir = run_dir / "harness"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_manifest, pdfs = create_input_manifest(args.dataset, run_dir)
    microbench_pdf = choose_microbench_pdf(pdfs)
    harness_config = run_dir / "harness_config.yaml"
    write_harness_config(
        path=harness_config,
        dataset_dir=input_manifest,
        harness_artifacts_dir=harness_artifacts_dir,
        preset=args.preset,
        embed_model_name=args.embed_model_name,
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "dataset": str(args.dataset.expanduser().resolve()),
        "input_manifest": str(input_manifest),
        "pdf_count": len(pdfs),
        "microbench_pdf": str(microbench_pdf),
        "environment": collect_environment(),
        "commands": [],
        "stage_results": {},
        "e2e_metrics": {},
        "validation_errors": [],
        "quality_checks": [
            {
                "name": "beir_recall",
                "status": "skipped",
                "detail": "bo20 has no query CSV / ground-truth recall config in this harness setup.",
            }
        ],
    }

    command_results: list[CommandResult] = []
    child_env = current_venv_env()
    benchmark_cli = python_main_command("nemo_retriever.tools.benchmark.__main__")
    harness_cli = python_main_command("nemo_retriever.harness")
    retriever = retriever_cli()
    summary["cli_entrypoints"] = {
        "benchmark": benchmark_cli,
        "harness": harness_cli,
        "retriever": retriever,
    }

    if not args.skip_stage:
        for repeat in range(1, args.stage_repeats + 1):
            stage_dir = stage_root / f"repeat_{repeat:02d}"
            command = [
                *benchmark_cli,
                "all",
                "run",
                "--pdf-path",
                str(microbench_pdf),
                "--output-dir",
                str(stage_dir),
                "--workers",
                args.stage_workers,
                "--batch-sizes",
                args.stage_batch_sizes,
                "--rows-split",
                str(args.rows_split),
                "--rows-extract",
                str(args.rows_extract),
                "--rows-page-elements",
                str(args.rows_page_elements),
                "--rows-ocr",
                str(args.rows_ocr),
                "--num-gpus",
                str(args.stage_num_gpus),
                "--num-cpus",
                str(args.stage_num_cpus),
            ]
            result = run_command(
                name=f"stage_microbench_{repeat:02d}",
                command=command,
                log_path=logs_dir / f"stage_microbench_{repeat:02d}.log",
                env=child_env,
                dry_run=args.dry_run,
            )
            command_results.append(result)
            stage_results = load_stage_results(stage_dir)
            summary["stage_results"][f"repeat_{repeat:02d}"] = stage_results
            if result.success and not args.dry_run and not stage_results:
                summary["validation_errors"].append(
                    f"{result.name} exited successfully but produced no stage JSON metrics under {stage_dir}"
                )

    successful_e2e_results: list[tuple[str, dict[str, Any]]] = []
    if not args.skip_e2e:
        e2e_plan: list[tuple[str, list[str]]] = []
        for repeat in range(1, args.e2e_repeats + 1):
            e2e_plan.append((f"e2e_fixed_{repeat:02d}", []))
        for repeat in range(1, args.heuristic_e2e_repeats + 1):
            e2e_plan.append((f"e2e_heuristic_{repeat:02d}", ["--override", "use_heuristics=true"]))

        for run_name, extra_args in e2e_plan:
            command = [
                *harness_cli,
                "run",
                "--config",
                str(harness_config),
                "--dataset",
                "bo20_local",
                "--preset",
                args.preset,
                "--run-name",
                run_name,
                "--tag",
                "dgx-spark",
                "--tag",
                "local-gpu",
                *extra_args,
            ]
            result = run_command(
                name=run_name,
                command=command,
                log_path=logs_dir / f"{run_name}.log",
                env=child_env,
                dry_run=args.dry_run,
            )
            command_results.append(result)
            harness_result = newest_harness_result(harness_artifacts_dir, run_name)
            if harness_result:
                metrics = dict(harness_result.get("summary_metrics") or {})
                metrics["success"] = bool(harness_result.get("success"))
                metrics["return_code"] = harness_result.get("return_code")
                metrics["artifact_dir"] = str((harness_artifacts_dir / sorted(
                    p.name for p in harness_artifacts_dir.glob(f"{run_name}_*")
                )[-1]).resolve()) if list(harness_artifacts_dir.glob(f"{run_name}_*")) else None
                summary["e2e_metrics"][run_name] = metrics
                if harness_result.get("success"):
                    successful_e2e_results.append((run_name, harness_result))
            elif result.success and not args.dry_run:
                summary["validation_errors"].append(
                    f"{run_name} exited successfully but produced no harness results.json under {harness_artifacts_dir}"
                )

    if successful_e2e_results and not args.skip_query_smoke:
        run_name, harness_result = successful_e2e_results[0]
        cfg = harness_result.get("test_config") or {}
        lancedb_uri = cfg.get("lancedb_uri")
        if lancedb_uri:
            command = [
                retriever,
                "query",
                args.query,
                "--lancedb-uri",
                str(lancedb_uri),
                "--table-name",
                "nv-ingest",
                "--top-k",
                "3",
                "--max-text-chars",
                "512",
            ]
            result = run_command(
                name=f"query_smoke_after_{run_name}",
                command=command,
                log_path=logs_dir / f"query_smoke_after_{run_name}.log",
                env=child_env,
                dry_run=args.dry_run,
            )
            command_results.append(result)
            summary["quality_checks"].append(
                {
                    "name": "query_smoke",
                    "status": "passed" if result.success else "failed",
                    "detail": f"Queried LanceDB table nv-ingest from {lancedb_uri}",
                }
            )

    summary["commands"] = [
        {
            "name": r.name,
            "command": r.command,
            "return_code": r.return_code,
            "elapsed_seconds": r.elapsed_seconds,
            "log_path": r.log_path,
        }
        for r in command_results
    ]
    embedding_quality = summarize_embedding_log_quality(command_results, summary["e2e_metrics"])
    summary["quality_checks"].append(embedding_quality)
    if embedding_quality["status"] == "failed":
        summary["validation_errors"].append(f"embedding_log_quality failed: {embedding_quality['detail']}")

    ingest_latencies = [
        float(metrics["ingest_secs"])
        for metrics in summary["e2e_metrics"].values()
        if metrics.get("ingest_secs") is not None
    ]
    pages_per_sec = [
        float(metrics["pages_per_sec_ingest"])
        for metrics in summary["e2e_metrics"].values()
        if metrics.get("pages_per_sec_ingest") is not None
    ]
    summary["latency_summary"] = {
        "ingest_secs": summarize_numeric(ingest_latencies),
        "pages_per_sec_ingest": summarize_numeric(pages_per_sec),
    }
    failed = [result for result in command_results if not result.success]
    summary["all_passed"] = not failed and not summary["validation_errors"]

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_markdown(summary, run_dir / "summary.md")
    write_e2e_csv(summary, run_dir / "e2e_metrics.csv")

    print(f"\nWrote benchmark summary: {summary_path}")
    return 1 if failed or summary["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
