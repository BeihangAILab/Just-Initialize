"""PCTSP representation, compression and local optimization."""
from __future__ import annotations

from ..core.representation import PreparedCompression
import math
import numpy as np
from dataclasses import dataclass, field
from itertools import pairwise
from numba import njit
from pathlib import Path
from typing import Any

pctsp_LEAF_SIZE = 8


pctsp_RESTORE_WINDOW = 24


pctsp_RESTORE_PASSES = 5


pctsp_AXIS_WINDOW = 16


pctsp_MAX_FIRST_INSERTIONS = 256


pctsp_MAX_SECOND_INSERTIONS = 128


pctsp_MAX_SECOND_ATTEMPTS = 512


pctsp_POLISH_WINDOW = 32


pctsp_POLISH_PASSES = 2


@dataclass(frozen=True)
class pctsp_PCTSPProblem:
    name: str
    points: np.ndarray
    prizes: np.ndarray
    penalties: np.ndarray
    quota: int

    @property
    def customers(self) -> int:
        return len(self.points) - 1

    def validate(self) -> None:
        count = len(self.points)
        if count < 2 or self.points.shape != (count, 2):
            raise ValueError("points must have shape (N+1, 2)")
        if self.prizes.shape != (count,) or self.penalties.shape != (count,):
            raise ValueError("prizes and penalties must have length N+1")
        if self.quota <= 0 or np.any(self.prizes < 0) or np.any(self.penalties < 0):
            raise ValueError("invalid quota, prizes, or penalties")
        if int(self.prizes[1:].sum()) < self.quota:
            raise ValueError("total available prize is below quota")


@dataclass
class pctsp_CompressionResult:
    decision: dict[str, Any]
    recovery: dict[str, Any]
    original_size: int
    decision_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class pctsp_OptimizeResult:
    cycle: np.ndarray
    moves: dict[str, Any] = field(default_factory=dict)


@njit(cache=False)
def pctsp_edge(points, left, right):

    dx = int(points[left, 0]) - int(points[right, 0])
    dy = int(points[left, 1]) - int(points[right, 1])
    return int(math.sqrt(dx * dx + dy * dy) + 0.5)


def pctsp_c_partition(points, customers):

    leaves, pending = [], [customers]
    while pending:
        nodes = pending.pop()
        if len(nodes) <= pctsp_LEAF_SIZE:
            leaves.append(nodes)
            continue
        coords = points[nodes]
        axis = int(np.argmax(np.ptp(coords, axis=0)))
        ordered = nodes[np.argsort(coords[:, axis], kind="stable")]
        middle = len(ordered) // 2
        pending.extend((ordered[:middle], ordered[middle:]))
    return leaves


def pctsp_c_two_opt_open(points, path):

    path = list(path)
    changed = True
    while changed:
        changed = False
        for i in range(len(path) - 3):
            for j in range(i + 2, len(path) - 1):
                old = pctsp_edge(points, path[i], path[i + 1]) + pctsp_edge(
                    points, path[j], path[j + 1]
                )
                new = pctsp_edge(points, path[i], path[j]) + pctsp_edge(
                    points, path[i + 1], path[j + 1]
                )
                if new < old:
                    path[i + 1 : j + 1] = reversed(path[i + 1 : j + 1])
                    changed = True
    return path


def pctsp_c_path_cost(points, path):

    return sum(pctsp_edge(points, left, right) for left, right in pairwise(path))


def pctsp_c_local_path(points, nodes):

    best = None
    for start in map(int, nodes):
        remaining = set(map(int, nodes))
        remaining.remove(start)
        path = [start]
        while remaining:
            current = path[-1]
            next_node = min(remaining, key=lambda node: pctsp_edge(points, current, node))
            path.append(next_node)
            remaining.remove(next_node)
        path = pctsp_c_two_opt_open(points, path)
        cost = pctsp_c_path_cost(points, path)
        if best is None or cost < best[0]:
            best = (cost, path)
    return best[1]


def pctsp_compress(problem):

    customers = np.arange(1, len(problem.points), dtype=np.int64)
    blocks = [
        pctsp_c_local_path(problem.points, leaf)
        for leaf in pctsp_c_partition(problem.points, customers)
    ]
    units = []
    for block in blocks:
        nodes = np.asarray(block, dtype=np.int64)
        units.append(
            {
                "prize": int(problem.prizes[nodes].sum()),
                "penalty": int(problem.penalties[nodes].sum()),
                "first": int(nodes[0]),
                "last": int(nodes[-1]),
                "center": problem.points[nodes].mean(axis=0).tolist(),
                "size": len(nodes),
            }
        )
    return pctsp_CompressionResult(
        {"unit_count": len(units), "units": units, "depot": problem.points[0].tolist()},
        {"units": blocks},
        problem.customers,
        len(units),
        {
            "method": "adaptive_kd_multistart_nearest_neighbour_2opt_fragments",
            "target_fragment": pctsp_LEAF_SIZE,
            "unit_count": len(units),
        },
    )


@njit(cache=False)
def pctsp_r_lower_bound(points, ordered, axis, value):

    low, high = 0, len(ordered)
    while low < high:
        middle = (low + high) // 2
        if points[ordered[middle], axis] < value:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=False)
def pctsp_r_spatial_2opt(points, route, ordered_x, ordered_y):

    count = len(route)
    position = np.full(len(points), -1, dtype=np.int64)
    for index in range(count):
        position[route[index]] = index
    moves = 0
    for _ in range(pctsp_RESTORE_PASSES):
        improved = False
        for index in range(count):
            left = route[index]
            right = route[(index + 1) % count]
            accepted = False
            for axis in range(2):
                ordered = ordered_x if axis == 0 else ordered_y
                at = pctsp_r_lower_bound(points, ordered, axis, points[left, axis])
                start = max(0, at - pctsp_RESTORE_WINDOW)
                stop = min(count, at + pctsp_RESTORE_WINDOW)
                for nearby_index in range(start, stop):
                    candidate = ordered[nearby_index]
                    candidate_index = position[candidate]
                    if candidate == left or candidate_index <= index + 1:
                        continue
                    if index == 0 and candidate_index == count - 1:
                        continue
                    following = route[(candidate_index + 1) % count]
                    old = pctsp_edge(points, left, right) + pctsp_edge(
                        points, candidate, following
                    )
                    new = pctsp_edge(points, left, candidate) + pctsp_edge(
                        points, right, following
                    )
                    if new < old:
                        lo, hi = index + 1, candidate_index
                        while lo < hi:
                            first, second = route[lo], route[hi]
                            route[lo], route[hi] = second, first
                            position[first], position[second] = hi, lo
                            lo += 1
                            hi -= 1
                        if lo == hi:
                            position[route[lo]] = lo
                        moves += 1
                        improved = True
                        accepted = True
                        break
                if accepted:
                    break
        if not improved:
            break
    return moves


def pctsp_o_nearby_route_nodes(points, x_order, y_order, candidate):

    candidates = set()
    for axis, ordered in ((0, x_order), (1, y_order)):
        at = int(np.searchsorted(points[ordered, axis], points[candidate, axis]))
        lo = max(0, at - pctsp_AXIS_WINDOW)
        hi = min(len(ordered), at + pctsp_AXIS_WINDOW)
        candidates.update(int(node) for node in ordered[lo:hi])
    return candidates


@njit(cache=False)
def pctsp_o_bounded_2opt(points, route):

    accepted = 0
    count = len(route)
    for _ in range(pctsp_POLISH_PASSES):
        improved = False
        for index in range(count - 2):
            left, right = route[index], route[index + 1]
            upper = min(count - 1, index + pctsp_POLISH_WINDOW)
            for candidate_index in range(index + 2, upper + 1):
                if index == 0 and candidate_index == count - 1:
                    continue
                candidate, following = (
                    route[candidate_index],
                    route[(candidate_index + 1) % count],
                )
                old = pctsp_edge(points, left, right) + pctsp_edge(
                    points, candidate, following
                )
                new = pctsp_edge(points, left, candidate) + pctsp_edge(
                    points, right, following
                )
                if new < old:
                    lo, hi = index + 1, candidate_index
                    while lo < hi:
                        route[lo], route[hi] = route[hi], route[lo]
                        lo += 1
                        hi -= 1
                    accepted += 1
                    improved = True
                    break
        if not improved:
            break
    return accepted


def pctsp_o_polish(problem, route):

    route = np.asarray(route, dtype=np.int64).copy()
    return route, int(pctsp_o_bounded_2opt(problem.points, route))


def pctsp_first_proposals(problem, route, selected):

    route_nodes = np.asarray(route, dtype=np.int64)
    x_order = route_nodes[np.argsort(problem.points[route_nodes, 0], kind="stable")]
    y_order = route_nodes[np.argsort(problem.points[route_nodes, 1], kind="stable")]
    position = np.full(len(problem.points), -1, dtype=np.int64)
    position[route_nodes] = np.arange(len(route_nodes), dtype=np.int64)
    proposals = []
    for customer in np.flatnonzero(~selected):
        if customer == 0:
            continue
        best_delta, best_anchor = None, None
        for anchor in pctsp_o_nearby_route_nodes(
            problem.points, x_order, y_order, int(customer)
        ):
            anchor_at = int(position[anchor])
            for left, right in (
                (route[anchor_at - 1], anchor),
                (anchor, route[(anchor_at + 1) % len(route)]),
            ):
                delta = int(
                    pctsp_edge(problem.points, left, int(customer))
                    + pctsp_edge(problem.points, int(customer), right)
                    - pctsp_edge(problem.points, left, right)
                )
                if best_delta is None or delta < best_delta:
                    best_delta, best_anchor = delta, (left, right)
        if best_delta is not None and int(problem.penalties[customer]) > best_delta:
            proposals.append(
                (
                    int(problem.penalties[customer]) - best_delta,
                    int(customer),
                    best_anchor,
                )
            )
    accepted = 0
    for _, customer, (left, right) in sorted(proposals, reverse=True)[
        :pctsp_MAX_FIRST_INSERTIONS
    ]:
        if accepted >= pctsp_MAX_FIRST_INSERTIONS:
            break
        try:
            left_at = route.index(left)
        except ValueError:
            continue
        if route[(left_at + 1) % len(route)] != right:
            continue
        delta = int(
            pctsp_edge(problem.points, left, customer)
            + pctsp_edge(problem.points, customer, right)
            - pctsp_edge(problem.points, left, right)
        )
        if int(problem.penalties[customer]) > delta:
            route.insert(left_at + 1, customer)
            selected[customer] = True
            accepted += 1
    return accepted


def pctsp_second_proposals(problem, route, selected):

    route_nodes = np.asarray(route, dtype=np.int64)
    x_order = route_nodes[np.argsort(problem.points[route_nodes, 0], kind="stable")]
    y_order = route_nodes[np.argsort(problem.points[route_nodes, 1], kind="stable")]
    position = np.full(len(problem.points), -1, dtype=np.int64)
    position[route_nodes] = np.arange(len(route_nodes), dtype=np.int64)
    proposals = []
    for customer in np.flatnonzero(~selected):
        if customer == 0:
            continue
        best_delta, anchor = None, None
        for candidate_anchor in pctsp_o_nearby_route_nodes(
            problem.points, x_order, y_order, int(customer)
        ):
            at = int(position[candidate_anchor])
            for left, right in (
                (route[at - 1], candidate_anchor),
                (candidate_anchor, route[(at + 1) % len(route)]),
            ):
                delta = int(
                    pctsp_edge(problem.points, left, int(customer))
                    + pctsp_edge(problem.points, int(customer), right)
                    - pctsp_edge(problem.points, left, right)
                )
                if best_delta is None or delta < best_delta:
                    best_delta, anchor = delta, left
        if best_delta is not None and int(problem.penalties[customer]) > best_delta:
            proposals.append(
                (int(problem.penalties[customer]) - best_delta, int(customer), anchor)
            )
    accepted = 0
    for _, customer, left in sorted(proposals, reverse=True)[:pctsp_MAX_SECOND_ATTEMPTS]:
        if accepted >= pctsp_MAX_SECOND_INSERTIONS:
            break
        left_at = route.index(left)
        right = route[(left_at + 1) % len(route)]
        delta = int(
            pctsp_edge(problem.points, left, customer)
            + pctsp_edge(problem.points, customer, right)
            - pctsp_edge(problem.points, left, right)
        )
        if int(problem.penalties[customer]) > delta:
            route.insert(left_at + 1, customer)
            selected[customer] = True
            accepted += 1
    return accepted


def pctsp_optimize(problem, cycle):

    route, first_2opt = pctsp_o_polish(problem, cycle[:-1])
    route = route.tolist()
    selected = np.zeros(len(problem.points), dtype=bool)
    selected[route] = True
    first_insertions = pctsp_first_proposals(problem, route, selected)
    route, second_2opt = pctsp_o_polish(problem, route)
    route = route.tolist()
    second_insertions = pctsp_second_proposals(problem, route, selected)
    route, third_2opt = pctsp_o_polish(problem, route)
    route = route.tolist()
    collected_prize = int(problem.prizes[np.asarray(route[1:], dtype=np.int64)].sum())
    removed = 0
    index = 1
    while index < len(route):
        customer = route[index]
        left = route[index - 1]
        right = route[index + 1] if index + 1 < len(route) else 0
        saved = int(
            pctsp_edge(problem.points, left, customer)
            + pctsp_edge(problem.points, customer, right)
            - pctsp_edge(problem.points, left, right)
        )
        if collected_prize - int(
            problem.prizes[customer]
        ) >= problem.quota and saved > int(problem.penalties[customer]):
            collected_prize -= int(problem.prizes[customer])
            selected[customer] = False
            route.pop(index)
            removed += 1
        else:
            index += 1
    route, fourth_2opt = pctsp_o_polish(problem, route)
    return pctsp_OptimizeResult(
        np.asarray([*route, 0], dtype=np.int64),
        {
            "accepted_moves": first_2opt + second_2opt + third_2opt + fourth_2opt,
            "first_pass_profitable_insertions": first_insertions,
            "second_pass_profitable_insertions": second_insertions,
            "quota_safe_deletions": removed,
            "method": (
                "jit_two_pass_insertion_current_anchor_quota_safe_deletion_then_2opt"
            ),
            "engine": "numba",
        },
    )


def pctsp_read_pctsp(path: str | Path) -> pctsp_PCTSPProblem:

    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    headers = {
        key.strip(): value.strip()
        for line in lines
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    if headers.get("TYPE") != "PC_TSP" or headers.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("only EUC_2D PC_TSP instances are supported")
    count = int(headers["DIMENSION"])
    sections = (
        "NODE_COORD_SECTION",
        "PRIZE_SECTION",
        "PENALTY_SECTION",
        "DEPOT_SECTION",
    )
    starts = {name: lines.index(name) + 1 for name in sections}

    def rows(name: str, next_name: str) -> list[str]:
        return [line for line in lines[starts[name] : starts[next_name] - 1] if line]

    points = np.zeros((count, 2), dtype=np.int64)
    prizes = np.zeros(count, dtype=np.int64)
    penalties = np.zeros(count, dtype=np.int64)
    for row in rows("NODE_COORD_SECTION", "PRIZE_SECTION"):
        index, x, y = row.split()
        points[int(index) - 1] = (int(x), int(y))
    for row in rows("PRIZE_SECTION", "PENALTY_SECTION"):
        index, value = row.split()
        prizes[int(index) - 1] = int(value)
    for row in rows("PENALTY_SECTION", "DEPOT_SECTION"):
        index, value = row.split()
        penalties[int(index) - 1] = int(value)
    if lines[starts["DEPOT_SECTION"]] != "1":
        raise ValueError("depot must be node 1")
    problem = pctsp_PCTSPProblem(
        headers.get("NAME", path.stem), points, prizes, penalties, int(headers["QUOTA"])
    )
    problem.validate()
    return problem



def compress(problem: object, seed: int = 0) -> PreparedCompression:
    """Build compact paths while preserving the original recovery representation."""
    result = pctsp_compress(problem)

    paths = [np.asarray(p, dtype=np.int64) for p in result.recovery["units"]]

    sources = list(range(len(paths)))

    return PreparedCompression(result, paths, sources)


def optimize(problem: object, compression: object, routes: list[np.ndarray], seed: int = 0):
    """Refine restored routes with the original problem-specific operators."""
    route = np.ascontiguousarray(routes[0], dtype=np.int64).copy()

    ordered_x = route[np.argsort(problem.points[route, 0], kind="stable")]

    ordered_y = route[np.argsort(problem.points[route, 1], kind="stable")]

    pctsp_r_spatial_2opt(problem.points, route, ordered_x, ordered_y)

    cycle = np.concatenate((np.array([0]), route, np.array([0])))

    return pctsp_optimize(problem, cycle).cycle



PCTSPProblem = pctsp_PCTSPProblem
read_pctsp = pctsp_read_pctsp
