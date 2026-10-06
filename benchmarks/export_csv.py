"""Export per-run and per-instance CSVs from saved benchmark JSON files.

No solver dependencies or reruns are required. Historic evaluation_time_s takes
precedence over time_s so all exported runtimes use the end-to-end boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIME_SCOPE = "loading_solving_validation"
RUN_COLUMNS = ("model", "problem", "suite", "scale", "instance", "repeat", "obj",
               "gap", "time_s", "evaluation_time_s", "solver_time_s", "time_scope",
               "valid", "reference", "reference_type", "error")
INSTANCE_COLUMNS = ("model", "problem", "suite", "scale", "instance", "obj", "avggap",
                    "avgtime", "avg_solver_time_s", "completed_runs", "valid_runs",
                    "expected_runs", "complete", "obj_min", "obj_max", "gap_std",
                    "time_std", "time_scope")


def normalize_run(raw: dict, source: Path) -> dict:
    """Keep archived solver-only timings separate from end-to-end timings."""
    row = dict(raw)
    if row.get("evaluation_time_s") is not None:
        elapsed = row["evaluation_time_s"]
        solver_time = row.get("solver_time_s")
        # The archived full benchmark used time_s for solver time, without a scope.
        if solver_time is None and not row.get("time_scope"):
            solver_time = row.get("time_s")
    else:
        if row.get("time_scope") not in (None, TIME_SCOPE):
            raise ValueError(f"{source}: no end-to-end runtime is available")
        # Historic standard evaluators already measured loading through validation.
        elapsed = row.get("time_s")
        solver_time = row.get("solver_time_s")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{source}: invalid or missing end-to-end runtime")
    if not isinstance(row.get("valid"), bool):
        raise ValueError(f"{source}: valid must be a boolean")
    for key in ("model", "problem", "suite", "instance"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{source}: missing {key}")
    for key in ("scale", "repeat"):
        if not isinstance(row.get(key), int) or row[key] < 1:
            raise ValueError(f"{source}: invalid {key}")
    objective = next((row[key] for key in ("obj", "objective", "score")
                      if row.get(key) is not None), None)
    gap = row.get("gap", row.get("gap_vs_reference_pct"))
    if row["valid"] and (not isinstance(objective, (int, float))
                         or not math.isfinite(objective)):
        raise ValueError(f"{source}: valid run has no finite objective")
    if gap is not None and (not isinstance(gap, (int, float)) or not math.isfinite(gap)):
        raise ValueError(f"{source}: invalid gap")
    if solver_time is not None and (not isinstance(solver_time, (int, float))
                                   or not math.isfinite(solver_time) or solver_time < 0):
        raise ValueError(f"{source}: invalid solver runtime")
    return {**row, "obj": objective, "gap": gap, "time_s": elapsed,
            "evaluation_time_s": elapsed, "solver_time_s": solver_time,
            "time_scope": TIME_SCOPE}


def mean(values: list) -> float | None:
    return statistics.mean(values) if values else None


def std(values: list) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export_results(results: Path, output: Path | None = None, *, model: str | None = None,
                   problem: str | None = None, expected_runs: int = 3) -> tuple[int, int]:
    if expected_runs < 1:
        raise ValueError("expected_runs must be positive")
    output = results if output is None else output
    rows = []
    seen = set()
    for path in sorted((results / "details").rglob("run_*.json")):
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if model is not None and raw.get("model") != model:
            continue
        if problem is not None and raw.get("problem") != problem:
            continue
        row = normalize_run(raw, path)
        key = tuple(row[k] for k in ("model", "problem", "suite", "scale", "instance", "repeat"))
        if key in seen:
            raise ValueError(f"{path}: duplicate instance/repetition {key}")
        seen.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("no matching run JSON files; existing exports were not changed")
    rows.sort(key=lambda row: tuple(row[k] for k in
                                   ("model", "problem", "suite", "scale", "instance", "repeat")))
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("model", "problem", "suite", "scale", "instance"))].append(row)
    summaries = []
    for key, runs in groups.items():
        valid = [r for r in runs if r["valid"]]
        objectives = [r["obj"] for r in valid]
        gaps = [r["gap"] for r in valid if r["gap"] is not None]
        times = [r["time_s"] for r in runs]
        solver_times = [r["solver_time_s"] for r in runs if r["solver_time_s"] is not None]
        summaries.append(dict(zip(INSTANCE_COLUMNS[:5], key, strict=True)) | {
            "obj": mean(objectives), "avggap": mean(gaps), "avgtime": mean(times),
            "avg_solver_time_s": mean(solver_times) if len(solver_times) == len(runs) else None,
            "completed_runs": len(runs), "valid_runs": len(valid), "expected_runs": expected_runs,
            "complete": {r["repeat"] for r in runs} == set(range(1, expected_runs + 1)),
            "obj_min": min(objectives) if objectives else None,
            "obj_max": max(objectives) if objectives else None,
            "gap_std": std(gaps), "time_std": std(times), "time_scope": TIME_SCOPE,
        })
    # Validate every input before touching either previous export.
    write_csv(output / "runs.csv", RUN_COLUMNS, rows)
    write_csv(output / "instance_summary.csv", INSTANCE_COLUMNS, summaries)
    return len(rows), len(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="output directory; defaults to --results")
    parser.add_argument("--model", help="export one result label")
    parser.add_argument("--problem", choices=("tsp", "cvrp", "pctsp", "vrptw"))
    parser.add_argument("--expected-runs", type=int, default=3)
    args = parser.parse_args()
    try:
        runs, instances = export_results(args.results, args.output, model=args.model,
                                         problem=args.problem, expected_runs=args.expected_runs)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(f"Exported {runs} runs and {instances} instances to {args.output or args.results}")


if __name__ == "__main__":
    main()
