"""Public api entry points."""
from __future__ import annotations
import numpy as np
from pathlib import Path

from .problems.tsp import tsp_TSPProblem
from .problems.cvrp import cvrp_CVRPProblem
from .problems.pctsp import pctsp_PCTSPProblem, pctsp_read_pctsp
from .problems.vrptw import vrptw_VRPTWProblem
from .core.pipeline import compress, solve, restore, optimize
from .core.oracle import Oracle
from .io import read_instance

CVRPProblem = cvrp_CVRPProblem


VRPTWProblem = vrptw_VRPTWProblem


PCTSPProblem = pctsp_PCTSPProblem


read_pctsp = pctsp_read_pctsp


def solve_tsp(points: np.ndarray, seed: int = 0) -> np.ndarray:
    problem = tsp_TSPProblem("standalone", np.asarray(points, dtype=np.float64), "EUC_2D")
    compact = compress("tsp", problem, seed)
    return optimize(compact, restore(solve(compact, Oracle(compact)), compact), seed)


def solve_cvrp(problem: cvrp_CVRPProblem) -> list[np.ndarray]:
    compact = compress("cvrp", problem)
    return optimize(compact, restore(solve(compact, Oracle(compact)), compact))


def solve_problem(problem: pctsp_PCTSPProblem) -> np.ndarray:
    compact = compress("pctsp", problem)
    return optimize(compact, restore(solve(compact, Oracle(compact)), compact))


def solve_vrptw(problem: vrptw_VRPTWProblem) -> list[np.ndarray]:
    compact = compress("vrptw", problem)
    return optimize(compact, restore(solve(compact, Oracle(compact)), compact))


def solve_instance(path: str | Path, seed: int = 0):

    kind, problem = read_instance(path)
    compact = compress(kind, problem, seed)
    oracle = Oracle(compact)
    return optimize(compact, restore(solve(compact, oracle), compact), seed)


