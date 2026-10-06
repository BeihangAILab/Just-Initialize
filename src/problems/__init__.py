"""Explicit registry of supported routing problem implementations."""
from types import MappingProxyType

from . import tsp, cvrp, pctsp, vrptw
from .base import ProblemAdapter

ADAPTERS = MappingProxyType({"tsp": tsp, "cvrp": cvrp, "pctsp": pctsp, "vrptw": vrptw})


def get_adapter(kind: str) -> ProblemAdapter:
    try:
        return ADAPTERS[kind]
    except KeyError:
        raise ValueError(f"unknown problem kind: {kind}") from None
