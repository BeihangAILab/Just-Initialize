"""四类算例的独立评测入口，规则移植自 HL/*/evaluation/locked/*_io.py。

仅在显式指定算例、套件或全部算例时求解。每例运行三遍，逐次保存完整解。
每个实例的可行性和目标值均在此独立复算，不采用 solver 自报值。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SOLVER = "src"
for _problem_module in ("cvrp_solver", "pctsp_solver", "tsp_solver", "vrptw_solver"):
    setattr(sys.modules[__name__], _problem_module,
            importlib.import_module(DEFAULT_SOLVER))

DATA = ROOT / "data"
RESULTS = ROOT / "results"
PROBLEMS = ("tsp", "cvrp", "pctsp", "vrptw")


def references(problem: str) -> dict[str, dict]:
    locations = {
        "tsp": ["tsp/bks/tsplib_bks.json", "tsp/bks/uniform_bks.json", "tsp/bks/extended_bks.json"],
        "cvrp": ["cvrp/bks/uniform_reference.json", "cvrp/bks/cvrplib_reference.json"],
        "pctsp": ["pctsp/uniform_reference.json", "pctsp/bks/uniform_reference.json"],
        "vrptw": ["vrptw/instances/uniform/uniform_reference.json",
                  "vrptw/bks/uniform_reference.json"],
    }
    rows = {}
    for location in locations[problem]:
        path = DATA / location
        if not path.is_file():
            continue
        for row in json.loads(path.read_text(encoding="utf-8"))["instances"]:
            rows[row.get("instance", row.get("name"))] = row
    return rows


def use_solver(problem: str, module) -> None:
    """Score one problem with an arbitrary solver module.

    The injected module doubles as the instance loader: every problem loader below
    goes through the same ``<problem>_solver`` global.
    """
    setattr(sys.modules[__name__], f"{problem}_solver", module)


def manifest(problem: str) -> dict[str, dict]:
    """读取 manifest，并补入实际目录中未登记的算例，兼容旧版或部分 manifest。"""
    rows: list[dict] = []
    for path in sorted((DATA / problem).glob("manifest*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    entries = scan_instances(problem)
    if not rows:
        return entries
    instance_root = DATA / problem / "instances" if problem == "tsp" else DATA / problem
    entries.update({
        row["instance"].strip(): {
            "path": instance_root / row["path"],
            "scale": int(row["dimension"]) if problem == "tsp" else int(row["customers"]),
            "suite": (
                row["instance"].split("_")[1] if row["category"] == "extended"
                else row["category"]
            ) if problem == "tsp" else Path(row["path"]).parts[1],
        }
        for row in rows
        if not (problem == "tsp" and row["category"] == "extended"
                and int(row["dimension"]) == 1000)
    })
    return entries


def scan_instances(problem: str) -> dict[str, dict]:
    """扫描实际算例目录，作为 manifest 的补充索引。"""
    root = DATA / problem / "instances"
    if not root.is_dir():
        return {}
    entries: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".tsp", ".vrp", ".pctsp"):
            continue
        name = path.stem
        parts = path.relative_to(root).parts
        suite = parts[0] if len(parts) > 1 else ""
        if problem == "tsp" and suite == "extended":
            suite = name.split("_")[1] if "_" in name else suite
        scale = scale_from_name(problem, name)
        if scale is None:
            scale = scale_from_file(path)
        if scale is None:
            continue
        # The published extended suite contains only 10k and 100k instances.
        if problem == "tsp" and parts[0] == "extended" and scale == 1000:
            continue
        entries[name] = {"path": path, "scale": scale, "suite": suite}
    return entries


def scale_from_name(problem: str, name: str) -> int | None:
    if problem == "tsp":
        match = re.match(r"tsp(\d+)_", name)
        return int(match.group(1)) if match else None
    match = re.search(r"-(\d+)k-", name)
    return int(match.group(1)) * 1000 if match else None


def scale_from_file(path: Path) -> int | None:
    """TSPLIB 头里的 DIMENSION（cvrp/vrptw 的客户数 = DIMENSION - 1）。"""
    try:
        text = path.read_text(encoding="ascii", errors="ignore")
    except OSError:
        return None
    match = re.search(r"DIMENSION\s*:\s*(\d+)", text)
    if not match:
        return None
    dimension = int(match.group(1))
    return dimension if path.suffix.lower() == ".tsp" else dimension - 1


def reference_fields(reference: dict | None, objective: int | None) -> dict:
    if reference is None:
        return {"reference": None, "reference_type": None, "gap_vs_reference_pct": None}
    value = reference.get("reference_value", reference.get("upper_bound"))
    value = int(value) if value is not None else None
    return {
        "reference": value,
        "reference_type": reference.get("reference_type"),
        "gap_vs_reference_pct": None if objective is None or not value else 100.0 * (objective - value) / value,
    }


def edge(a: np.ndarray, b: np.ndarray, kind: str = "EUC_2D") -> int:
    distance = math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    if kind == "EUC_2D":
        return math.floor(distance + 0.5)
    if kind == "CEIL_2D":
        return math.ceil(distance)
    raise ValueError(f"unsupported edge weight type: {kind}")


def _section(lines: list[str], start: str, end: str) -> list[str]:
    return [line for line in lines[lines.index(start) + 1:lines.index(end)] if line]


def load_cvrp(path: Path) -> cvrp_solver.CVRPProblem:
    """按 HL/cvrp-hl-main/evaluation/locked/cvrp_io.py 读取 CVRPLIB。"""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    headers = {key.strip(): value.strip() for line in lines[:lines.index("NODE_COORD_SECTION")]
               if ":" in line for key, value in [line.split(":", 1)]}
    coordinates = {int(parts[0]): (float(parts[1]), float(parts[2]))
                   for row in _section(lines, "NODE_COORD_SECTION", "DEMAND_SECTION")
                   for parts in [row.split()]}
    demands = {int(parts[0]): int(parts[1])
               for row in _section(lines, "DEMAND_SECTION", "DEPOT_SECTION")
               for parts in [row.split()]}
    depot = int(next(row for row in lines[lines.index("DEPOT_SECTION") + 1:]
                     if row and row != "-1"))
    customers = sorted(node for node in coordinates if node != depot)
    capacity = int(headers["CAPACITY"])
    if headers.get("EDGE_WEIGHT_TYPE", "EUC_2D") != "EUC_2D":
        raise ValueError("CVRP only supports EUC_2D")
    if capacity <= 0 or any(not 0 < demands[node] <= capacity for node in customers):
        raise ValueError("invalid CVRP capacity or demand")
    return cvrp_solver.CVRPProblem(
        name=headers.get("NAME", path.stem),
        points=np.asarray([coordinates[node] for node in customers], dtype=np.float64),
        demands=np.asarray([demands[node] for node in customers], dtype=np.int64),
        depot=np.asarray(coordinates[depot], dtype=np.float64),
        capacity=capacity, vehicle_limit=None, edge_weight_type="EUC_2D",
    )


def load_vrptw(path: Path) -> vrptw_solver.VRPTWProblem:
    """按 HL/vrptw-hl-main/evaluation/locked/vrptw_io.py 读取 CVRPTW。"""
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    headers = {key.strip(): value.strip() for line in lines[:lines.index("NODE_COORD_SECTION")]
               if ":" in line for key, value in [line.split(":", 1)]}
    if headers.get("TYPE") != "CVRPTW" or headers.get("EDGE_WEIGHT_TYPE", "EUC_2D") != "EUC_2D":
        raise ValueError("only EUC_2D CVRPTW is supported")
    dimension = int(headers["DIMENSION"])
    points = np.zeros((dimension, 2), dtype=np.float64)
    demands = np.zeros(dimension, dtype=np.int64)
    tw_start = np.zeros(dimension, dtype=np.int64)
    tw_end = np.zeros(dimension, dtype=np.int64)
    service = np.zeros(dimension, dtype=np.int64)
    for row in _section(lines, "NODE_COORD_SECTION", "DEMAND_SECTION"):
        node, x, y = row.split()
        points[int(node) - 1] = (float(x), float(y))
    for row in _section(lines, "DEMAND_SECTION", "DEPOT_SECTION"):
        node, demand = row.split()
        demands[int(node) - 1] = int(demand)
    for row in _section(lines, "TIME_WINDOW_SECTION", "SERVICE_TIME_SECTION"):
        node, start, end = row.split()
        tw_start[int(node) - 1] = int(start)
        tw_end[int(node) - 1] = int(end)
    for row in lines[lines.index("SERVICE_TIME_SECTION") + 1:]:
        if row == "EOF":
            break
        if row:
            node, duration = row.split()
            service[int(node) - 1] = int(duration)
    depots = [row for row in _section(lines, "DEPOT_SECTION", "TIME_WINDOW_SECTION") if row != "-1"]
    if depots != ["1"]:
        raise ValueError("VRPTW depot must be node 1")
    return vrptw_solver.VRPTWProblem(
        name=headers.get("NAME", path.stem), points=points, demands=demands,
        capacity=int(headers["CAPACITY"]), tw_start=tw_start, tw_end=tw_end,
        service_time=service, vehicle_limit=None,
    )


def tsp(path: Path) -> dict:
    # HL/tsp-hl-main/evaluation/locked/tsp_io.py: load_problem, validate_tour, tour_length
    headers = {}
    coordinates = []
    in_coordinates = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "NODE_COORD_SECTION":
            in_coordinates = True
        elif line == "EOF":
            break
        elif in_coordinates and line:
            _, x, y = line.split()[:3]
            coordinates.append((float(x), float(y)))
        elif ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    points = np.asarray(coordinates, dtype=np.float64)
    dimension = int(headers["DIMENSION"])
    if points.shape != (dimension, 2):
        raise ValueError("TSP coordinate count does not match DIMENSION")
    tour = np.asarray(tsp_solver.solve_tsp(points, seed=0))
    valid = bool(tour.ndim == 1 and len(tour) == dimension
                 and np.issubdtype(tour.dtype, np.integer)
                 and tour.min(initial=0) >= 0 and tour.max(initial=-1) < dimension
                 and np.unique(tour).size == dimension)
    objective = None
    if valid:
        kind = headers.get("EDGE_WEIGHT_TYPE", "EUC_2D")
        objective = sum(edge(points[int(a)], points[int(b)], kind)
                        for a, b in zip(tour, np.roll(tour, -1), strict=True))
    return {"valid": valid, "error": "" if valid else "tour must visit every node exactly once",
            "objective": objective, "tour_length": int(len(tour)),
            "solution": {"tour": tour.tolist()}}


def cvrp(path: Path) -> dict:
    # HL/cvrp-hl-main/evaluation/locked/cvrp_io.py: validate_routes, route_cost
    problem = load_cvrp(path)
    routes = cvrp_solver.solve_cvrp(problem)
    valid = isinstance(routes, (list, tuple)) and bool(routes)
    error = ""
    if valid:
        valid = all(isinstance(route, np.ndarray) and route.ndim == 1 and len(route)
                    and np.issubdtype(route.dtype, np.integer) for route in routes)
        if not valid:
            error = "routes must be nonempty integer arrays"
    else:
        error = "solution must contain routes"
    if valid:
        visited = np.concatenate(routes)
        valid = (len(visited) == len(problem.points)
                 and visited.min(initial=0) >= 0 and visited.max(initial=-1) < len(problem.points)
                 and np.unique(visited).size == len(problem.points))
        if not valid:
            error = "customers must be visited exactly once"
    if valid and any(int(problem.demands[route].sum()) > problem.capacity for route in routes):
        valid, error = False, "capacity exceeded"
    objective = None
    if valid:
        objective = 0
        for route in routes:
            previous = problem.depot
            for node in route:
                current = problem.points[int(node)]
                objective += edge(previous, current)
                previous = current
            objective += edge(previous, problem.depot)
    return {"valid": bool(valid), "error": error, "objective": objective,
            "route_count": len(routes),
            "route_loads": [int(problem.demands[route].sum()) for route in routes] if valid else None,
            "solution": {"routes": [route.tolist() for route in routes]}}


def pctsp(path: Path) -> dict:
    # HL/pctsp-hl-main/evaluation/locked/pctsp_io.py: metrics
    problem = pctsp_solver.read_pctsp(path)
    cycle = np.asarray(pctsp_solver.solve_problem(problem))
    valid = (cycle.ndim == 1 and cycle.size >= 3 and np.issubdtype(cycle.dtype, np.integer)
             and cycle[0] == 0 and cycle[-1] == 0)
    error = "" if valid else "cycle must start and end at depot"
    if valid:
        selected = cycle[1:-1]
        valid = bool(np.all((selected >= 1) & (selected < len(problem.points)))
                     and np.unique(selected).size == len(selected))
        if not valid:
            error = "invalid or duplicate customer"
    if valid:
        prize = int(problem.prizes[selected].sum())
        if prize < problem.quota:
            valid, error = False, "quota not met"
    objective = None
    if valid:
        distance = sum(edge(problem.points[int(a)], problem.points[int(b)])
                       for a, b in zip(cycle[:-1], cycle[1:], strict=True))
        omitted = np.ones(len(problem.points), dtype=bool)
        omitted[0] = False
        omitted[selected] = False
        penalty = int(problem.penalties[omitted].sum())
        objective = distance + penalty
    return {"valid": bool(valid), "error": error, "objective": objective,
            "tour_length": distance if valid else None,
            "unvisited_penalty": penalty if valid else None,
            "collected_prize": prize if valid else None,
            "selected_count": len(selected) if valid else None,
            "solution": {"cycle": cycle.tolist()}}


def vrptw(path: Path) -> dict:
    # HL/vrptw-hl-main/evaluation/locked/vrptw_io.py: validate_solution, route_metrics
    problem = load_vrptw(path)
    routes = vrptw_solver.solve_vrptw(problem)
    valid = isinstance(routes, (list, tuple)) and bool(routes)
    error = "" if valid else "solution must contain routes"
    if valid:
        valid = all(isinstance(route, np.ndarray) and route.ndim == 1 and len(route)
                    and np.issubdtype(route.dtype, np.integer) for route in routes)
        if not valid:
            error = "routes must be nonempty integer arrays"
    if valid:
        visited = np.concatenate(routes)
        valid = (len(visited) == len(problem.points) - 1 and visited.min(initial=1) >= 1
                 and visited.max(initial=-1) < len(problem.points)
                 and np.unique(visited).size == len(visited))
        if not valid:
            error = "customers must be visited exactly once"
    objective = 0 if valid else None
    loads, finishes = [], []
    if valid:
        for route in routes:
            load, clock, previous = 0, int(problem.tw_start[0]), 0
            for node in route:
                node = int(node)
                travel = edge(problem.points[previous], problem.points[node])
                objective += travel
                clock = max(int(problem.tw_start[node]), clock + travel)
                if clock > problem.tw_end[node]:
                    valid, error = False, "customer time window exceeded"
                clock += int(problem.service_time[node])
                load += int(problem.demands[node])
                previous = node
            travel = edge(problem.points[previous], problem.points[0])
            objective += travel
            clock += travel
            if load > problem.capacity:
                valid, error = False, "capacity exceeded"
            if clock > problem.tw_end[0]:
                valid, error = False, "depot return time exceeded"
            loads.append(load)
            finishes.append(clock)
    return {"valid": bool(valid), "error": error, "objective": objective if valid else None,
            "route_count": len(routes), "route_loads": loads if valid else None,
            "max_route_finish_time": max(finishes) if valid else None,
            "solution": {"routes": [route.tolist() for route in routes]}}


EVALUATORS = {"tsp": tsp, "cvrp": cvrp, "pctsp": pctsp, "vrptw": vrptw}
REPEATS = 3


def detail_path(model: str, problem: str, suite: str, instance: str, repeat: int,
                legacy: bool = False) -> Path:
    """Run file of one (model, problem, instance, repeat); the model is the folder.

    ``legacy`` addresses the pre-model layout (results/details/<problem>/...), which
    only ever held the default ``<problem>_solver`` rows.
    """
    if legacy:
        return RESULTS / "details" / problem / suite / instance / f"run_{repeat}.json"
    return RESULTS / "details" / model / problem / suite / instance / f"run_{repeat}.json"


def existing_detail(model: str, problem: str, suite: str, instance: str,
                    repeat: int) -> Path | None:
    """A run already done, in either layout, still counts as done."""
    current = detail_path(model, problem, suite, instance, repeat)
    if current.is_file():
        return current
    legacy = detail_path(model, problem, suite, instance, repeat, legacy=True)
    if (model == f"{problem}_solver" and legacy.is_file()
            and json.loads(legacy.read_text(encoding="utf-8")).get("model") == model):
        return legacy
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    try:
        temporary.replace(path)
    except PermissionError:
        latest = path.with_name(path.stem + "_latest.csv")
        temporary.replace(latest)
        print(f"{path.name} is in use; wrote {latest.name}", flush=True)


def write_summary() -> None:
    groups: dict[tuple[str, str, str, int], list[dict]] = defaultdict(list)
    for path in (RESULTS / "details").rglob("run_*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        groups[(row["model"], row["problem"], row["suite"], row["scale"])].append(row)
    # 期望次数按 manifest 的算例数算，因此同一算例集下的多个模型可以并列汇总。
    manifest_runs: dict[tuple[str, str, int], int] = defaultdict(int)
    for problem in PROBLEMS:
        for item in manifest(problem).values():
            manifest_runs[(problem, item["suite"], item["scale"])] += REPEATS
    columns = ("model", "problem", "suite", "scale", "avg_score", "mean_gap", "mean_time_s",
               "completed_runs", "valid_runs", "expected_runs")
    summary_rows = []
    active_problems = {key[1] for key in groups}
    keys = set(groups)
    for key in sorted(keys):
        rows = groups[key]
        valid = [row for row in rows if row["valid"] and row["score"] is not None]
        gaps = [row["gap"] for row in valid if row["gap"] is not None]
        summary_rows.append(dict(zip(columns[:4], key, strict=True)) | {
            "avg_score": statistics.mean(row["score"] for row in valid) if valid else "",
            "mean_gap": statistics.mean(gaps) if gaps else "",
            "mean_time_s": statistics.mean(row["time_s"] for row in rows) if rows else "",
            "completed_runs": len(rows), "valid_runs": len(valid),
            "expected_runs": manifest_runs[(key[1], key[2], key[3])],
        })
    write_csv(RESULTS / "summary.csv", columns, summary_rows)
    by_suite: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_suite[(row["problem"], row["suite"])].append(row)
    for (problem, suite), rows in by_suite.items():
        write_csv(RESULTS / "summary" / problem / f"{suite}.csv", columns, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, choices=PROBLEMS)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--instance", action="append", help="算例名，可重复指定")
    selection.add_argument("--suite", help="数据集名称，例如 uniform")
    selection.add_argument("--all", action="store_true", help="运行全部算例")
    parser.add_argument("--scale", type=int, help="只运行指定节点规模")
    parser.add_argument("--solver", help=f"求解器模块名，默认 {DEFAULT_SOLVER}；"
                                         "对照其它实现时指定模块名")
    parser.add_argument("--model", help="结果里的模型标签，默认用求解器模块名；"
                                        "重跑同一求解器到新硬件时用它区分")
    args = parser.parse_args()

    model = args.model or args.solver or DEFAULT_SOLVER
    if args.solver:
        use_solver(args.problem, importlib.import_module(args.solver))
    entries = manifest(args.problem)
    if args.suite:
        names = sorted((name for name, item in entries.items() if item["suite"] == args.suite),
                       key=lambda name: (entries[name]["scale"], name))
        if not names:
            parser.error(f"unknown or empty suite: {args.suite}")
    elif args.all:
        names = sorted(entries, key=lambda name: (entries[name]["suite"], entries[name]["scale"], name))
    else:
        names = args.instance
    if args.scale is not None:
        names = [name for name in names if name in entries and entries[name]["scale"] == args.scale]
        if not names:
            parser.error(f"no instances at scale {args.scale}")
    missing = [name for name in names if name not in entries]
    if missing:
        parser.error(f"unknown instances: {', '.join(missing)}")
    refs = references(args.problem)
    pending = [name for name in names if any(
        existing_detail(model, args.problem, entries[name]["suite"], name, repeat) is None
        for repeat in range(1, REPEATS + 1))]
    if pending:
        # 首次 numba 编译不计入任何一次正式计时；预热结果不落盘。
        print(f"warmup: {pending[0]}", flush=True)
        EVALUATORS[args.problem](entries[pending[0]]["path"])
    for index, name in enumerate(names):
        item = entries[name]
        for repeat in range(1, REPEATS + 1):
            done = existing_detail(model, args.problem, item["suite"], name, repeat)
            if done is not None:
                print(f"skip existing: {name} run {repeat}", flush=True)
                continue
            output = detail_path(model, args.problem, item["suite"], name, repeat)
            started = time.perf_counter()
            try:
                result = EVALUATORS[args.problem](item["path"])
            except Exception as exc:
                result = {"valid": False, "error": f"{type(exc).__name__}: {exc}",
                          "objective": None, "solution": None}
            elapsed = time.perf_counter() - started
            reference = reference_fields(refs.get(name), result["objective"])
            row = {"model": model, "problem": args.problem,
                   "suite": item["suite"], "scale": item["scale"], "instance": name,
                   "repeat": repeat, "score": result["objective"],
                   "gap": reference["gap_vs_reference_pct"], "time_s": elapsed,
                   **result, **reference}
            write_json(output, row)
            print(f"{name} run {repeat}/{REPEATS}: "
                  f"{'valid' if row['valid'] else 'INVALID'}; score={row['score']}; "
                  f"time={elapsed:.2f}s", flush=True)
        if index + 1 == len(names) or entries[names[index + 1]]["scale"] != item["scale"]:
            write_summary()
    failed = False
    for name in names:
        for repeat in range(1, REPEATS + 1):
            done = existing_detail(model, args.problem, entries[name]["suite"], name, repeat)
            if done is None or not json.loads(done.read_text(encoding="utf-8"))["valid"]:
                failed = True
    print(f"saved: {RESULTS / 'summary.csv'}; details: {RESULTS / 'details' / args.problem}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
