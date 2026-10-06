"""Small offline checks for readers, four solvers, validation, timing, and CSV export.

Uses temporary synthetic files; never changes the datasets or saved experiments.
First-use Numba compilation can make the initial invocation take longer.
"""
from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import evaluate
import export_csv
import run_parallel
import src as ji


def write_fixtures(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(2026)
    points = np.vstack(([500, 500], rng.integers(0, 1000, size=(64, 2))))
    coordinates = "\n".join(f"{i} {x} {y}" for i, (x, y) in enumerate(points, 1))
    paths = {}
    for kind, suffix, label in [("tsp", ".tsp", "TSP"), ("cvrp", ".vrp", "CVRP"),
                                ("pctsp", ".pctsp", "PC_TSP"), ("vrptw", ".vrp", "CVRPTW")]:
        lines = [f"NAME : smoke_{kind}", f"TYPE : {label}", "DIMENSION : 65",
                 "EDGE_WEIGHT_TYPE : EUC_2D"]
        if kind in ("cvrp", "vrptw"):
            lines += ["CAPACITY : 20"]
        if kind == "pctsp":
            lines += ["QUOTA : 100"]
        lines += ["NODE_COORD_SECTION", coordinates]
        if kind in ("cvrp", "vrptw"):
            lines += ["DEMAND_SECTION", "1 0"]
            lines += [f"{i} 2" for i in range(2, 66)]
            lines += ["DEPOT_SECTION", "1", "-1"]
        if kind == "pctsp":
            lines += ["PRIZE_SECTION", "1 0"] + [f"{i} 5" for i in range(2, 66)]
            lines += ["PENALTY_SECTION", "1 0"] + [f"{i} 200" for i in range(2, 66)]
            lines += ["DEPOT_SECTION", "1", "-1"]
        if kind == "vrptw":
            lines += ["TIME_WINDOW_SECTION"] + [f"{i} 0 100000" for i in range(1, 66)]
            lines += ["SERVICE_TIME_SECTION"] + [f"{i} 2" for i in range(1, 66)]
        # Also exercise the supported TSPLIB variant without an EOF terminator.
        if kind != "tsp":
            lines += ["EOF"]
        path = root / f"smoke_{kind}{suffix}"
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        paths[kind] = path
    return paths


def check_export(root: Path) -> None:
    results = root / "export"
    for repeat, elapsed in enumerate((2.0, 4.0, 6.0), 1):
        row = {"model": "smoke", "problem": "tsp", "suite": "uniform", "scale": 65,
               "instance": "fixture", "repeat": repeat, "objective": 100 + repeat,
               "gap": float(repeat), "valid": True, "reference": 100,
               "time_s": elapsed / 2, "evaluation_time_s": elapsed}
        if repeat == 3:
            row.update(time_s=elapsed, time_scope=export_csv.TIME_SCOPE)
        evaluate.write_json(results / "details" / f"run_{repeat}.json", row)
    assert export_csv.export_results(results) == (3, 1)
    with (results / "instance_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["avgtime"]) == 4 and float(row["obj"]) == 102
    assert row["avg_solver_time_s"] == "" and row["complete"] == "True"
    (results / "details" / "run_3.json").unlink()
    export_csv.export_results(results)
    with (results / "instance_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["complete"] == "False"
    payload = (results / "details" / "run_1.json").read_text(encoding="utf-8")
    (results / "details" / "duplicate.json").write_text(payload, encoding="utf-8")
    (results / "details" / "duplicate.json").rename(results / "details" / "run_duplicate.json")
    try:
        export_csv.export_results(results)
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate runs must not be counted twice")
    print("PASS CSV: historic/new timings, three-run statistics, incomplete and duplicate runs")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ji_smoke_") as directory:
        root = Path(directory)
        paths = write_fixtures(root)
        for kind, path in paths.items():
            detected, _ = ji.read_instance(path)
            assert detected == kind
            warmup = evaluate.EVALUATORS[kind](path)
            assert warmup["valid"], (kind, warmup["error"])
            result = evaluate.timed_evaluation(kind, path)
            assert result["valid"], (kind, result["error"])
            assert result["objective"] is not None and math.isfinite(result["objective"])
            assert result["time_s"] == result["evaluation_time_s"] >= 0
            assert result["time_scope"] == evaluate.TIME_SCOPE
            print(f"PASS {kind}: reading, solving, independent validation, end-to-end timing", flush=True)
        with patch.object(evaluate.tsp_solver, "solve_tsp", return_value=np.zeros(65, dtype=int)):
            assert not evaluate.tsp(paths["tsp"])["valid"]
        with patch.dict(evaluate.EVALUATORS, {"tsp": lambda path: (_ for _ in ()).throw(ValueError("fixture failure"))}):
            failure = evaluate.timed_evaluation("tsp", paths["tsp"])
            assert not failure["valid"] and "fixture failure" in failure["error"]
            assert failure["time_s"] >= 0
        print("PASS invalid solution and failed-evaluation recording")
        # Exercise the worker's persisted timing fields without launching a full suite.
        with patch.object(evaluate, "RESULTS", root / "parallel"):
            entries = {"fixture": {"path": paths["tsp"], "suite": "smoke", "scale": 65}}
            run_parallel._initialize_worker("tsp", "smoke", entries, {}, "src", None)
            _, completed, valid, _ = run_parallel._run_instance("fixture")
            assert completed == valid == 3
            row = json.loads(evaluate.detail_path("smoke", "tsp", "smoke", "fixture", 1).read_text())
            assert row["time_scope"] == export_csv.TIME_SCOPE
            assert row["time_s"] == row["evaluation_time_s"]
            evaluate.write_summary()
        with patch.object(evaluate, "RESULTS", root / "sequential"), \
                patch.object(evaluate, "manifest", side_effect=lambda kind: entries if kind == "tsp" else {}), \
                patch.object(evaluate, "references", return_value={}), \
                patch("sys.argv", ["evaluate.py", "--problem", "tsp", "--instance", "fixture", "--model", "smoke"]):
            evaluate.main()
        print("PASS sequential CLI and parallel worker timing records")
        check_export(root)
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
