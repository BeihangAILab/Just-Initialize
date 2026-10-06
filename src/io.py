"""Public io entry points."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from .problems.tsp import tsp_TSPProblem
from .problems.cvrp import cvrp_CVRPProblem
from .problems.pctsp import pctsp_read_pctsp
from .problems.vrptw import vrptw_VRPTWProblem

def detect_problem(path: str | Path) -> str:

    path = Path(path)
    if path.suffix.lower() == ".tsp":
        return "tsp"
    if path.suffix.lower() == ".pctsp":
        return "pctsp"
    if path.suffix.lower() != ".vrp":
        raise ValueError(f"unsupported instance suffix: {path.suffix}")
    with path.open(encoding="ascii") as handle:
        for line in handle:
            if line.strip() == "NODE_COORD_SECTION":
                break
            if line.partition(":")[0].strip() == "TYPE":
                value = line.partition(":")[2].strip().upper()
                if value == "CVRPTW":
                    return "vrptw"
                if value == "CVRP":
                    return "cvrp"
    raise ValueError(".vrp header must declare TYPE CVRP or CVRPTW")


def _section(lines: list[str], start: str, end: str) -> list[str]:
    return [line for line in lines[lines.index(start) + 1:lines.index(end)] if line]


def read_instance(path: str | Path):
    path = Path(path)
    kind = detect_problem(path)
    if kind == "pctsp":
        return kind, pctsp_read_pctsp(path)
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    headers = {key.strip(): value.strip()
               for line in lines[:lines.index("NODE_COORD_SECTION")]
               if ":" in line for key, value in [line.split(":", 1)]}
    if headers.get("EDGE_WEIGHT_TYPE", "EUC_2D") != "EUC_2D":
        raise ValueError("only EUC_2D is supported")
    if kind == "tsp":
        rows = _section(lines, "NODE_COORD_SECTION", "EOF")
        points = np.asarray([(float(row.split()[1]), float(row.split()[2]))
                             for row in rows], dtype=np.float64)
        if len(points) != int(headers["DIMENSION"]):
            raise ValueError("TSP dimension does not match coordinates")
        return kind, tsp_TSPProblem(headers.get("NAME", path.stem), points, "EUC_2D")
    coords = {int(parts[0]): (float(parts[1]), float(parts[2]))
              for row in _section(lines, "NODE_COORD_SECTION", "DEMAND_SECTION")
              for parts in [row.split()]}
    demands = {int(parts[0]): int(parts[1])
               for row in _section(lines, "DEMAND_SECTION", "DEPOT_SECTION")
               for parts in [row.split()]}
    if kind == "cvrp":
        depot = int(next(row for row in lines[lines.index("DEPOT_SECTION") + 1:]
                         if row and row != "-1"))
        customers = sorted(node for node in coords if node != depot)
        return kind, cvrp_CVRPProblem(
            headers.get("NAME", path.stem),
            np.asarray([coords[node] for node in customers], dtype=np.float64),
            np.asarray([demands[node] for node in customers], dtype=np.int64),
            np.asarray(coords[depot], dtype=np.float64), int(headers["CAPACITY"]),
            None, "EUC_2D",
        )
    dimension = int(headers["DIMENSION"])
    points = np.asarray([coords[node] for node in range(1, dimension + 1)], dtype=np.float64)
    demand_array = np.asarray([demands[node] for node in range(1, dimension + 1)], dtype=np.int64)
    windows = {int(parts[0]): (int(parts[1]), int(parts[2]))
               for row in _section(lines, "TIME_WINDOW_SECTION", "SERVICE_TIME_SECTION")
               for parts in [row.split()]}
    service = {int(parts[0]): int(parts[1])
               for row in _section(lines, "SERVICE_TIME_SECTION", "EOF")
               for parts in [row.split()]}
    depot_rows = [row for row in _section(lines, "DEPOT_SECTION", "TIME_WINDOW_SECTION")
                  if row != "-1"]
    if depot_rows != ["1"]:
        raise ValueError("VRPTW depot must be node 1")
    return kind, vrptw_VRPTWProblem(
        headers.get("NAME", path.stem), points, demand_array,
        int(headers["CAPACITY"]),
        np.asarray([windows[node][0] for node in range(1, dimension + 1)], dtype=np.int64),
        np.asarray([windows[node][1] for node in range(1, dimension + 1)], dtype=np.int64),
        np.asarray([service[node] for node in range(1, dimension + 1)], dtype=np.int64),
        None,
    )


