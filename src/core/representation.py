"""Shared compact representation and search parameters."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Unit:
    nodes: np.ndarray
    center: np.ndarray
    demand: int = 0
    prize: int = 0
    penalty: int = 0
    source: int = 0


@dataclass
class Compact:
    kind: str
    problem: object
    units: list[Unit]
    source: object
    seed: int = 0


@dataclass(frozen=True)
class ProblemSpec:

    single_route: bool = False
    optional_nodes: bool = False
    prize_quota: bool = False
    capacity_limited: bool = False


SPECS: dict[str, ProblemSpec] = {
    "tsp": ProblemSpec(single_route=True),
    "cvrp": ProblemSpec(capacity_limited=True),
    "pctsp": ProblemSpec(single_route=True, optional_nodes=True, prize_quota=True),
    "vrptw": ProblemSpec(capacity_limited=True),
}


def problem_spec(kind: str) -> ProblemSpec:

    try:
        return SPECS[kind]
    except KeyError:
        raise ValueError(f"no problem spec for kind: {kind}") from None


SEARCH_NEIGHBORS = 24


MERGE_NEIGHBORS = 32


SEARCH_SEED = 0


DOWNSTREAM_SHORTLIST = 2
DOWNSTREAM_SEARCH_ROUNDS = 4
DOWNSTREAM_SAMPLES = 1


VRPTW_STARTS = 2


@dataclass(frozen=True)
class Item:
    unit: int
    reverse: bool = False


@dataclass
class Plan:
    routes: list[list[Item]]



@dataclass
class PreparedCompression:
    """Problem adapter output before shared unit features are assembled."""

    source: object
    paths: list[np.ndarray]
    sources: list[int]
