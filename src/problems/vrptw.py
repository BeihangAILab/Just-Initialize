"""VRPTW representation, compression and local optimization."""
from __future__ import annotations

from ..core.representation import PreparedCompression, SEARCH_SEED, VRPTW_STARTS
from ..core.recovery import edge_distance
import heapq
import math
import numpy as np
import time
from dataclasses import dataclass, field
from numba import njit
from typing import Any

vrptw_GRID_DIVISIONS = 96


vrptw_COLLECTION_CAP = 64


vrptw_UNIT_NEIGHBOR_CAP = 8


vrptw_BUILDER_CANDIDATE_K = 48


vrptw_LOOKAHEAD_WEIGHT = 0.5


vrptw_ROUTES_PER_UNIT = 40


vrptw_LOCAL_FRAGMENT_MAX_NODES = 16


vrptw_UNIT_NEIGHBORS = 32


vrptw_WINDOW_BUCKETS = 8


vrptw_RESTORE_ROUNDS = 5


vrptw_RESTORE_DROPS = 5


vrptw_RESTORE_SOURCES = 16


vrptw_RESTORE_POLISH_PASSES = 1


vrptw_POLISH_CANDIDATE_K = 12


vrptw_OPTIMIZE_MOVES = 4000


vrptw_OPTIMIZE_PASSES = 18


vrptw_OPTIMIZE_CANDIDATE_K = 64


vrptw_OPTIMIZE_MAX_PAIRS = 200_000


vrptw_OPTIMIZE_SECONDS = 2.0

EVALUATION_ROUTE_PAIRS = 32
EVALUATION_CROSS_MOVES = 8


@dataclass(frozen=True)
class vrptw_VRPTWProblem:
    name: str
    points: np.ndarray
    demands: np.ndarray
    capacity: int
    tw_start: np.ndarray
    tw_end: np.ndarray
    service_time: np.ndarray
    vehicle_limit: int | None = None

    @property
    def customers(self) -> int:
        return len(self.points) - 1


@dataclass
class vrptw_CompressionResult:
    decision: Any
    recovery: Any
    original_size: int
    decision_size: int
    representation_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class vrptw_OptimizeResult:
    routes: list[np.ndarray]
    moves: dict[str, int] = field(default_factory=dict)


def vrptw_list_coords(points: np.ndarray) -> tuple[list[float], list[float]]:

    return points[:, 0].tolist(), points[:, 1].tolist()


def vrptw_list_ints(array: np.ndarray) -> list[int]:

    return [int(value) for value in array]


@njit(cache=False)
def vrptw_edge(coords: np.ndarray, left: int, right: int) -> int:

    dx = coords[left, 0] - coords[right, 0]
    dy = coords[left, 1] - coords[right, 1]
    return int(np.floor(np.sqrt(dx * dx + dy * dy) + 0.5))


@njit(cache=False)
def vrptw_build_candidates(points: np.ndarray, k: int, grid: int, cap: int) -> np.ndarray:

    count = points.shape[0]
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)
    cell_x = np.empty(count, dtype=np.int64)
    cell_y = np.empty(count, dtype=np.int64)
    cell_of = np.empty(count, dtype=np.int64)
    for i in range(count):
        cx = int((points[i, 0] - min_x) * grid / span_x)
        cy = int((points[i, 1] - min_y) * grid / span_y)
        if cx >= grid:
            cx = grid - 1
        if cy >= grid:
            cy = grid - 1
        cell_x[i] = cx
        cell_y[i] = cy
        cell_of[i] = cy * grid + cx
    order = np.argsort(cell_of).astype(np.int64)
    start = np.zeros(grid * grid + 1, dtype=np.int64)
    counts = np.zeros(grid * grid, dtype=np.int64)
    for i in range(count):
        counts[cell_of[i]] += 1
    position = 0
    for cell in range(grid * grid):
        start[cell] = position
        position += counts[cell]
    start[grid * grid] = count
    candidates = np.empty((count, k), dtype=np.int64)
    buffer = np.empty(cap, dtype=np.int64)
    distances = np.empty(cap, dtype=np.float64)
    for i in range(count):
        cx = cell_x[i]
        cy = cell_y[i]
        px = points[i, 0]
        py = points[i, 1]
        filled = 0
        ring = 0
        while filled < cap and ring < grid:
            x0 = max(cx - ring, 0)
            x1 = min(cx + ring, grid - 1)
            y0 = max(cy - ring, 0)
            y1 = min(cy + ring, grid - 1)
            for gy in range(y0, y1 + 1):
                for gx in range(x0, x1 + 1):
                    if filled >= cap:
                        break
                    on_ring = gx == x0 or gx == x1 or gy == y0 or gy == y1
                    if not on_ring:
                        continue
                    cell = gy * grid + gx
                    for index in range(start[cell], start[cell + 1]):
                        if filled >= cap:
                            break
                        node = order[index]
                        if node == i:
                            continue
                        dx = points[node, 0] - px
                        dy = points[node, 1] - py
                        buffer[filled] = node
                        distances[filled] = dx * dx + dy * dy
                        filled += 1
            ring += 1
        for a in range(filled):
            best = a
            for b in range(a + 1, filled):
                if distances[b] < distances[best]:
                    best = b
            if best != a:
                temporary = distances[a]
                distances[a] = distances[best]
                distances[best] = temporary
                temporary_node = buffer[a]
                buffer[a] = buffer[best]
                buffer[best] = temporary_node
        for slot in range(k):
            candidates[i, slot] = buffer[slot] if slot < filled else -1
    return candidates


@njit(cache=False)
def vrptw_simulate_route(
    route: np.ndarray,
    coords: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
) -> tuple[bool, int, int]:

    time = tw_start[0]
    cost = 0
    previous = 0
    feasible = True
    for index in range(route.shape[0]):
        node = route[index]
        travel = vrptw_edge(coords, previous, node)
        cost += travel
        time += travel
        if time < tw_start[node]:
            time = tw_start[node]
        if time > tw_end[node]:
            feasible = False
        time += service[node]
        previous = node
    travel = vrptw_edge(coords, previous, 0)
    cost += travel
    time += travel
    if time > tw_end[0]:
        feasible = False
    return feasible, cost, time


@njit(cache=False)
def vrptw_constrained_routes(
    coords: np.ndarray,
    demands: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    capacity: int,
    candidates: np.ndarray,
    seed_order: np.ndarray,
    allowed: np.ndarray,
    scope_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    count = coords.shape[0]
    width = candidates.shape[1]
    scope_count = scope_nodes.shape[0]
    used = np.zeros(count, dtype=np.uint8)
    nodes = np.empty(count, dtype=np.int64)
    offsets = np.empty(count + 1, dtype=np.int64)
    output = 0
    routes = 0
    horizon = tw_end[0]
    for order_index in range(count - 1):
        seed = seed_order[order_index]
        if used[seed] or allowed[seed] == 0:
            continue
        offsets[routes] = output
        nodes[output] = seed
        output += 1
        used[seed] = 1
        load = demands[seed]
        previous = seed
        arrival = tw_start[0] + vrptw_edge(coords, 0, seed)
        start = arrival if arrival > tw_start[seed] else tw_start[seed]
        time = start + service[seed]
        while True:
            best = -1
            best_score = 1e18
            best_time = 0
            for slot in range(width):
                node = candidates[previous, slot]
                if node <= 0 or used[node] or allowed[node] == 0:
                    continue
                if load + demands[node] > capacity:
                    continue
                travel = vrptw_edge(coords, previous, node)
                arrival = time + travel
                wait = tw_start[node] - arrival if arrival < tw_start[node] else 0
                start = arrival if arrival > tw_start[node] else tw_start[node]
                if start > tw_end[node]:
                    continue
                after = start + service[node]
                if after + vrptw_edge(coords, node, 0) > horizon:
                    continue
                score = travel + wait
                if score < best_score:
                    best_score = score
                    best = node
                    best_time = after
            if best < 0:
                for index in range(scope_count):
                    node = scope_nodes[index]
                    if used[node]:
                        continue
                    if load + demands[node] > capacity:
                        continue
                    travel = vrptw_edge(coords, previous, node)
                    arrival = time + travel
                    wait = tw_start[node] - arrival if arrival < tw_start[node] else 0
                    start = arrival if arrival > tw_start[node] else tw_start[node]
                    if start > tw_end[node]:
                        continue
                    after = start + service[node]
                    if after + vrptw_edge(coords, node, 0) > horizon:
                        continue
                    score = travel + wait
                    if score < best_score:
                        best_score = score
                        best = node
                        best_time = after
            if best < 0:
                break
            nodes[output] = best
            output += 1
            used[best] = 1
            load += demands[best]
            previous = best
            time = best_time
        routes += 1
    offsets[routes] = output
    return nodes[:output], offsets[: routes + 1]


@njit(cache=False)
def vrptw_constrained_routes_beam(
    coords: np.ndarray,
    demands: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    capacity: int,
    candidates: np.ndarray,
    seed_order: np.ndarray,
    allowed: np.ndarray,
    scope_nodes: np.ndarray,
    max_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:

    count = coords.shape[0]
    width = candidates.shape[1]
    scope_count = scope_nodes.shape[0]
    used = np.zeros(count, dtype=np.uint8)
    nodes = np.empty(count, dtype=np.int64)
    offsets = np.empty(count + 1, dtype=np.int64)
    output = 0
    routes = 0
    horizon = tw_end[0]
    for order_index in range(count - 1):
        seed = seed_order[order_index]
        if used[seed] or allowed[seed] == 0:
            continue
        offsets[routes] = output
        nodes[output] = seed
        output += 1
        used[seed] = 1
        load = demands[seed]
        previous = seed
        arrival = tw_start[0] + vrptw_edge(coords, 0, seed)
        start = tw_start[seed] if tw_start[seed] > arrival else arrival
        current_time = start + service[seed]
        while True:
            if output - offsets[routes] >= max_nodes:
                break
            chosen = -1
            chosen_score = 1e18
            limit = width
            for pass_index in range(2):
                limit = width if pass_index == 0 else scope_count
                for slot in range(limit):
                    node = (
                        candidates[previous, slot]
                        if pass_index == 0
                        else scope_nodes[slot]
                    )
                    if node <= 0 or used[node] or allowed[node] == 0:
                        continue
                    if load + demands[node] > capacity:
                        continue
                    arrival = current_time + vrptw_edge(coords, previous, node)
                    start = tw_start[node] if tw_start[node] > arrival else arrival
                    if (
                        start > tw_end[node]
                        or start + service[node] + vrptw_edge(coords, node, 0) > horizon
                    ):
                        continue
                    score = float(
                        vrptw_edge(coords, previous, node)
                        + (tw_start[node] - arrival if tw_start[node] > arrival else 0)
                    )
                    next_best = 1e18
                    for next_slot in range(width):
                        nxt = candidates[node, next_slot]
                        if nxt <= 0 or nxt == node or used[nxt] or allowed[nxt] == 0:
                            continue
                        if load + demands[node] + demands[nxt] > capacity:
                            continue
                        next_arrival = start + service[node] + vrptw_edge(coords, node, nxt)
                        next_start = (
                            tw_start[nxt]
                            if tw_start[nxt] > next_arrival
                            else next_arrival
                        )
                        if (
                            next_start > tw_end[nxt]
                            or next_start + service[nxt] + vrptw_edge(coords, nxt, 0)
                            > horizon
                        ):
                            continue
                        next_score = float(
                            vrptw_edge(coords, node, nxt)
                            + (
                                tw_start[nxt] - next_arrival
                                if tw_start[nxt] > next_arrival
                                else 0
                            )
                        )
                        if next_score < next_best:
                            next_best = next_score
                    lookahead = 0.0 if next_best >= 1e17 else next_best
                    total = score + vrptw_LOOKAHEAD_WEIGHT * lookahead
                    if total < chosen_score:
                        chosen = node
                        chosen_score = total
                if chosen >= 0:
                    break
            if chosen < 0:
                break
            arrival = current_time + vrptw_edge(coords, previous, chosen)
            start = tw_start[chosen] if tw_start[chosen] > arrival else arrival
            nodes[output] = chosen
            output += 1
            used[chosen] = 1
            load += demands[chosen]
            previous = chosen
            current_time = start + service[chosen]
        routes += 1
    offsets[routes] = output
    return nodes[:output], offsets[: routes + 1]


@njit(cache=False)
def vrptw_best_relocate(
    a: np.ndarray,
    b: np.ndarray,
    coords: np.ndarray,
    demands: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    capacity: int,
    candidates: np.ndarray,
    scratch_a: np.ndarray,
    scratch_b: np.ndarray,
) -> tuple[int, int, int, int, int]:

    feasible_a, cost_a, load_a = vrptw_route_metrics_seq(
        a, a.shape[0], coords, demands, tw_start, tw_end, service, capacity
    )
    feasible_b, cost_b, load_b = vrptw_route_metrics_seq(
        b, b.shape[0], coords, demands, tw_start, tw_end, service, capacity
    )
    if not (feasible_a and feasible_b):
        return 0, 0, 0, 0, -1
    best_delta = 0
    best_from_a = 0
    best_position = 0
    best_insert = 0
    best_node = -1
    for direction in range(2):
        source = a if direction == 0 else b
        target = b if direction == 0 else a
        source_cost = cost_a if direction == 0 else cost_b
        target_cost = cost_b if direction == 0 else cost_a
        target_load = load_b if direction == 0 else load_a
        width = candidates.shape[1]
        for position in range(source.shape[0]):
            node = source[position]
            if target_load + demands[node] > capacity:
                continue
            length = 0
            for index in range(source.shape[0]):
                if index != position:
                    scratch_a[length] = source[index]
                    length += 1
            if length == 0:
                continue
            new_source_feasible, new_source_cost, _ = vrptw_route_metrics_seq(
                scratch_a, length, coords, demands, tw_start, tw_end, service, capacity
            )
            if not new_source_feasible:
                continue
            for slot in range(width):
                anchor = candidates[node, slot]
                if anchor <= 0:
                    continue
                anchor_position = -1
                for index in range(target.shape[0]):
                    if target[index] == anchor:
                        anchor_position = index
                        break
                if anchor_position < 0:
                    continue
                for insert in (anchor_position, anchor_position + 1):
                    length_b = 0
                    for index in range(target.shape[0] + 1):
                        if index == insert:
                            scratch_b[length_b] = node
                            length_b += 1
                        if index < target.shape[0]:
                            scratch_b[length_b] = target[index]
                            length_b += 1
                    new_target_feasible, new_target_cost, _ = vrptw_route_metrics_seq(
                        scratch_b,
                        length_b,
                        coords,
                        demands,
                        tw_start,
                        tw_end,
                        service,
                        capacity,
                    )
                    if not new_target_feasible:
                        continue
                    delta = (new_source_cost + new_target_cost) - (
                        source_cost + target_cost
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_from_a = direction
                        best_position = position
                        best_insert = insert
                        best_node = node
    return best_from_a, best_position, best_insert, best_delta, best_node


@njit(cache=False)
def vrptw_best_tail_exchange(
    a: np.ndarray,
    b: np.ndarray,
    coords: np.ndarray,
    demands: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    capacity: int,
    scratch_a: np.ndarray,
    scratch_b: np.ndarray,
) -> tuple[int, int, int]:

    feasible_a, cost_a, _ = vrptw_route_metrics_seq(
        a, a.shape[0], coords, demands, tw_start, tw_end, service, capacity
    )
    feasible_b, cost_b, _ = vrptw_route_metrics_seq(
        b, b.shape[0], coords, demands, tw_start, tw_end, service, capacity
    )
    if not (feasible_a and feasible_b):
        return -1, -1, 0
    best_delta = 0
    best_i = -1
    best_j = -1
    for cut_a in range(-1, a.shape[0]):
        for cut_b in range(-1, b.shape[0]):
            length_a = 0
            for index in range(cut_a + 1):
                scratch_a[length_a] = a[index]
                length_a += 1
            for index in range(cut_b + 1, b.shape[0]):
                scratch_a[length_a] = b[index]
                length_a += 1
            length_b = 0
            for index in range(cut_b + 1):
                scratch_b[length_b] = b[index]
                length_b += 1
            for index in range(cut_a + 1, a.shape[0]):
                scratch_b[length_b] = a[index]
                length_b += 1
            if length_a == 0 or length_b == 0:
                continue
            new_a_feasible, new_a_cost, _ = vrptw_route_metrics_seq(
                scratch_a,
                length_a,
                coords,
                demands,
                tw_start,
                tw_end,
                service,
                capacity,
            )
            if not new_a_feasible:
                continue
            new_b_feasible, new_b_cost, _ = vrptw_route_metrics_seq(
                scratch_b,
                length_b,
                coords,
                demands,
                tw_start,
                tw_end,
                service,
                capacity,
            )
            if not new_b_feasible:
                continue
            delta = (new_a_cost + new_b_cost) - (cost_a + cost_b)
            if delta < best_delta:
                best_delta = delta
                best_i = cut_a
                best_j = cut_b
    return best_i, best_j, best_delta


def vrptw_to_routes(nodes: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:

    return [
        np.ascontiguousarray(nodes[offsets[i] : offsets[i + 1]], dtype=np.int64)
        for i in range(len(offsets) - 1)
    ]


def vrptw_build_routes(problem: vrptw_VRPTWProblem) -> list[list[int]]:

    points = np.ascontiguousarray(problem.points, dtype=np.float64)
    demands = np.ascontiguousarray(problem.demands, dtype=np.int64)
    service = np.ascontiguousarray(problem.service_time, dtype=np.int64)
    tw_start = np.ascontiguousarray(problem.tw_start, dtype=np.int64)
    tw_end = np.ascontiguousarray(problem.tw_end, dtype=np.int64)
    capacity = int(problem.capacity)
    candidates = vrptw_build_candidates(points, vrptw_POLISH_CANDIDATE_K, vrptw_GRID_DIVISIONS, vrptw_COLLECTION_CAP)
    customers = int(problem.customers)
    allowed = np.ones(len(points), dtype=np.uint8)
    allowed[0] = 0
    scope_nodes = np.arange(1, customers + 1, dtype=np.int64)
    seeds = np.ascontiguousarray(
        np.lexsort((np.arange(1, customers + 1), tw_start[1:])) + 1, dtype=np.int64
    )
    nodes, offsets = vrptw_constrained_routes(
        points,
        demands,
        tw_start,
        tw_end,
        service,
        capacity,
        candidates,
        seeds,
        allowed,
        scope_nodes,
    )
    built = [list(route) for route in vrptw_to_routes(nodes, offsets)]
    if not built:
        raise ValueError("no construction policy produced a route set")
    return built


def vrptw_build_routes_lookahead(
    problem: vrptw_VRPTWProblem, candidate_k: int
) -> list[list[int]]:

    points = np.ascontiguousarray(problem.points, dtype=np.float64)
    demands = np.ascontiguousarray(problem.demands, dtype=np.int64)
    service = np.ascontiguousarray(problem.service_time, dtype=np.int64)
    tw_start = np.ascontiguousarray(problem.tw_start, dtype=np.int64)
    tw_end = np.ascontiguousarray(problem.tw_end, dtype=np.int64)
    capacity = int(problem.capacity)
    candidates = vrptw_build_candidates(points, candidate_k, vrptw_GRID_DIVISIONS, vrptw_COLLECTION_CAP)
    customers = int(problem.customers)
    allowed = np.ones(len(points), dtype=np.uint8)
    allowed[0] = 0
    scope_nodes = np.arange(1, customers + 1, dtype=np.int64)
    seeds = np.ascontiguousarray(
        np.lexsort((np.arange(1, customers + 1), tw_start[1:])) + 1, dtype=np.int64
    )
    nodes, offsets = vrptw_constrained_routes_beam(
        points,
        demands,
        tw_start,
        tw_end,
        service,
        capacity,
        candidates,
        seeds,
        allowed,
        scope_nodes,
        vrptw_LOCAL_FRAGMENT_MAX_NODES,
    )
    built = [list(route) for route in vrptw_to_routes(nodes, offsets)]
    if not built:
        raise ValueError("no construction policy produced a route set")
    return built


def vrptw_group_routes(
    routes: list[list[int]],
    per_unit: int,
    node_neighbors: np.ndarray,
    xs: list[float],
    ys: list[float],
    size: int,
) -> list[list[int]]:

    count = len(routes)
    chains: dict[int, list[list[int]]] = {
        index: [routes[index]] for index in range(count)
    }
    heads: dict[int, int] = {index: routes[index][0] for index in range(count)}
    tails: dict[int, int] = {index: routes[index][-1] for index in range(count)}
    owner = np.full(size, -1, dtype=np.int64)
    for index, route in enumerate(routes):
        owner[route] = index
    alive = [True] * count
    parent = list(range(count))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def leg(left: int, right: int) -> int:
        return math.floor(math.hypot(xs[right] - xs[left], ys[right] - ys[left]) + 0.5)

    heap: list[tuple[int, int, int, bool]] = []

    def push(chain: int) -> None:
        for neighbor in node_neighbors[tails[chain]]:
            if neighbor < 0:
                continue
            other = int(owner[neighbor])
            if other < 0:
                continue
            other = root(other)
            if other == chain or not alive[other]:
                continue
            if len(chains[chain]) + len(chains[other]) > per_unit:
                continue
            for flip in (False, True):
                entry = tails[other] if flip else heads[other]
                saving = leg(0, tails[chain]) + leg(0, entry) - leg(tails[chain], entry)
                if saving > 0:
                    heapq.heappush(heap, (-saving, chain, other, flip))

    for index in range(count):
        push(index)
    while heap:
        negative, first, second, flip = heapq.heappop(heap)
        first, second = root(first), root(second)
        if first == second or not alive[first] or not alive[second]:
            continue
        if len(chains[first]) + len(chains[second]) > per_unit:
            continue
        entry = tails[second] if flip else heads[second]
        if leg(0, tails[first]) + leg(0, entry) - leg(tails[first], entry) != -negative:
            continue
        if flip:
            chains[second] = [route[::-1] for route in reversed(chains[second])]
            heads[second], tails[second] = tails[second], heads[second]
        chains[first] = chains[first] + chains[second]
        tails[first] = tails[second]
        alive[second] = False
        parent[second] = first
        for route in chains[second]:
            for node in route:
                owner[node] = first
        del chains[second]
        push(first)
    return [chains[index] for index in range(count) if alive[index]]


def vrptw_compress(
    problem: vrptw_VRPTWProblem,
    builder=None,
    routes_per_unit: int = vrptw_ROUTES_PER_UNIT,
    candidate_k: int = vrptw_POLISH_CANDIDATE_K,
) -> vrptw_CompressionResult:

    points = np.ascontiguousarray(problem.points, dtype=np.float64)
    demands = np.ascontiguousarray(problem.demands, dtype=np.int64)
    service = np.ascontiguousarray(problem.service_time, dtype=np.int64)
    tw_start = np.ascontiguousarray(problem.tw_start, dtype=np.int64)
    tw_end = np.ascontiguousarray(problem.tw_end, dtype=np.int64)
    horizon = float(tw_end[0] - tw_start[0])
    xs, ys = vrptw_list_coords(points)
    service_list = vrptw_list_ints(service)
    tw_start_list = vrptw_list_ints(tw_start)
    tw_end_list = vrptw_list_ints(tw_end)
    depot = points[0]

    routes = (builder or vrptw_build_routes)(problem)
    node_neighbors = vrptw_build_candidates(
        points, vrptw_UNIT_NEIGHBOR_CAP, vrptw_GRID_DIVISIONS, vrptw_COLLECTION_CAP
    )
    units_routes = vrptw_group_routes(
        routes, routes_per_unit, node_neighbors, xs, ys, len(points)
    )
    units = len(units_routes)
    if units < 1:
        raise ValueError("compression produced no units")

    sequences: list[list[int]] = []
    sub_offsets = [0]
    sub_head: list[int] = []
    sub_tail: list[int] = []
    sub_size: list[int] = []
    sub_demand: list[int] = []
    sub_occ: list[int] = []
    sub_release: list[int] = []
    sub_latest: list[int] = []
    unit_route_count = np.zeros(units, dtype=np.int64)
    for index, sub_routes in enumerate(units_routes):
        for route in sub_routes:
            head, tail = int(route[0]), int(route[-1])
            summary = vrptw_fragment_summary(
                xs, ys, tw_start_list, tw_end_list, service_list, route
            )
            if summary[0] < 0:
                summary = vrptw_fragment_summary(
                    xs, ys, tw_start_list, tw_end_list, service_list, route[::-1]
                )
            sub_head.append(head)
            sub_tail.append(tail)
            sub_size.append(len(route))
            sub_demand.append(int(demands[route].sum()))
            sub_occ.append(max(summary[0], 0))
            sub_release.append(max(summary[1], 0))
            sub_latest.append(max(summary[2], 0))
            sequences.append(list(route))
        unit_route_count[index] = len(sub_routes)
        sub_offsets.append(len(sequences))

    per_unit_nodes = [
        np.asarray([node for route in sub_routes for node in route], dtype=np.int64)
        for sub_routes in units_routes
    ]
    head = np.asarray([route[0][0] for route in units_routes], dtype=np.int64)
    tail = np.asarray([route[-1][-1] for route in units_routes], dtype=np.int64)
    sizes = np.asarray([len(nodes) for nodes in per_unit_nodes], dtype=np.int64)

    def unit_summary(nodes: np.ndarray, reverse: bool):
        order = list(nodes[::-1]) if reverse else list(nodes)
        summary = vrptw_fragment_summary(
            xs, ys, tw_start_list, tw_end_list, service_list, order
        )
        return summary

    summaries = [unit_summary(nodes, False) for nodes in per_unit_nodes]
    reversed_summaries = [unit_summary(nodes, True) for nodes in per_unit_nodes]
    occupancy = np.asarray([max(item[0], 0) for item in summaries], dtype=np.int64)
    release = np.asarray([max(item[1], 0) for item in summaries], dtype=np.int64)
    latest = np.asarray(
        [item[2] if item[0] >= 0 else 0 for item in summaries], dtype=np.int64
    )
    occupancy_rev = np.asarray(
        [max(item[0], 0) for item in reversed_summaries], dtype=np.int64
    )
    release_rev = np.asarray(
        [max(item[1], 0) for item in reversed_summaries], dtype=np.int64
    )
    latest_rev = np.asarray(
        [item[2] if item[0] >= 0 else 0 for item in reversed_summaries], dtype=np.int64
    )

    unit_service = np.asarray(
        [int(service[nodes].sum()) for nodes in per_unit_nodes], dtype=np.int64
    )
    depot_distance = np.asarray(
        [
            float(
                np.hypot(points[nodes, 0] - depot[0], points[nodes, 1] - depot[1]).min()
            )
            for nodes in per_unit_nodes
        ]
    )
    usable = np.maximum(horizon - 2.0 * depot_distance, 1.0)
    vehicle_lb = np.maximum(1, np.ceil(unit_service / usable)).astype(np.int64)
    slack = np.maximum(0, latest - tw_start[head])

    profile = np.zeros((units, vrptw_WINDOW_BUCKETS), dtype=np.int64)
    for index, nodes in enumerate(per_unit_nodes):
        bucket = np.minimum(
            (
                (tw_start[nodes] - tw_start[0]) / max(horizon, 1.0) * vrptw_WINDOW_BUCKETS
            ).astype(np.int64),
            vrptw_WINDOW_BUCKETS - 1,
        )
        np.add.at(profile[index], bucket, 1)

    endpoints = np.empty((2 * units, 2), dtype=np.float64)
    endpoints[0::2] = points[head]
    endpoints[1::2] = points[tail]
    end_neighbors = vrptw_build_candidates(
        endpoints, vrptw_UNIT_NEIGHBORS, vrptw_GRID_DIVISIONS, vrptw_COLLECTION_CAP
    )
    end_unit = np.repeat(np.arange(units, dtype=np.int64), 2)

    node_unit = np.full(len(points), -1, dtype=np.int64)
    for index, nodes in enumerate(per_unit_nodes):
        node_unit[nodes] = index

    decision = {
        "unit_count": int(units),
        "head": head,
        "tail": tail,
        "unit_demands": np.asarray(
            [int(demands[nodes].sum()) for nodes in per_unit_nodes], dtype=np.int64
        ),
        "unit_sizes": sizes,
        "unit_service": unit_service,
        "unit_occupancy": occupancy,
        "unit_release": release,
        "unit_latest_entry": latest,
        "unit_occupancy_rev": occupancy_rev,
        "unit_release_rev": release_rev,
        "unit_latest_entry_rev": latest_rev,
        "unit_slack": slack.astype(np.int64),
        "unit_vehicle_lb": vehicle_lb,
        "unit_route_count": unit_route_count,
        "unit_window_profile": profile,
        "end_neighbors": end_neighbors,
        "end_unit": end_unit,
        "sub_offsets": np.asarray(sub_offsets, dtype=np.int64),
        "sub_head": np.asarray(sub_head, dtype=np.int64),
        "sub_tail": np.asarray(sub_tail, dtype=np.int64),
        "sub_size": np.asarray(sub_size, dtype=np.int64),
        "sub_demand": np.asarray(sub_demand, dtype=np.int64),
        "sub_occupancy": np.asarray(sub_occ, dtype=np.int64),
        "sub_release": np.asarray(sub_release, dtype=np.int64),
        "sub_latest_entry": np.asarray(sub_latest, dtype=np.int64),
    }
    recovery = {
        "sequences": [np.asarray(sequence, dtype=np.int64) for sequence in sequences],
        "sub_offsets": np.asarray(sub_offsets, dtype=np.int64),
        "node_unit": node_unit,
        "candidates": vrptw_build_candidates(
            points, candidate_k, vrptw_GRID_DIVISIONS, vrptw_COLLECTION_CAP
        ),
        "source_routes": len(routes),
    }
    representation_bytes = int(
        sum(
            value.nbytes for value in decision.values() if isinstance(value, np.ndarray)
        )
    )
    metadata = {
        "family": "path.v9",
        "units": int(units),
        "source_routes": len(routes),
        "routes_per_unit": routes_per_unit,
        "seed_policy": "tw_start",
        "constructor": "time_driven_insertion_single_policy",
        "customers": int(problem.customers),
        "compression_ratio": float(problem.customers / units),
        "mean_unit_size": float(sizes.mean()),
        "mean_routes_per_unit": float(unit_route_count.mean()),
        "mean_unit_slack": float(slack.mean()),
        "vehicle_lb_sum": int(vehicle_lb.sum()),
        "vehicle_lb_note": (
            "ceil(service_sum / (H - 2*min depot distance)); valid lower "
            "bound assuming each unit is served by routes contained in it"
        ),
        "decision_features": [
            "unit_portal",
            "unit_size",
            "unit_demand",
            "time_summary",
            "forward_time_slack",
            "vehicle_lower_bound",
            "window_profile",
            "sub_route_table",
        ],
        "change": "B1: multi-vehicle units plus a time-aware decision interface",
    }
    return vrptw_CompressionResult(
        decision=decision,
        recovery=recovery,
        original_size=int(len(points)),
        decision_size=int(units),
        representation_bytes=representation_bytes,
        metadata=metadata,
    )


def vrptw_fragment_summary(
    xs: list[float],
    ys: list[float],
    tw_start: list[int],
    tw_end: list[int],
    service: list[int],
    order: list[int],
) -> tuple[int, int, int]:

    def leg(left: int, right: int) -> int:
        return math.floor(math.hypot(xs[right] - xs[left], ys[right] - ys[left]) + 0.5)

    occupancy = 0
    for index in range(1, len(order)):
        occupancy += leg(order[index - 1], order[index])
    occupancy += sum(service[node] for node in order)

    if len(order) == 1:
        node = order[0]
        if tw_start[node] > tw_end[node]:
            return -1, -1, -1
        return occupancy, tw_start[node] + service[node], tw_end[node]

    time = float(tw_start[order[0]])
    if time > tw_end[order[0]]:
        return -1, -1, -1
    time += service[order[0]]
    for index in range(1, len(order)):
        node = order[index]
        time += leg(order[index - 1], node)
        start = tw_start[node]
        if time < start:
            time = start
        if time > tw_end[node]:
            return -1, -1, -1
        time += service[node]
    release = int(time)

    latest = float(tw_end[order[-1]])
    for index in range(len(order) - 2, -1, -1):
        node = order[index]
        travel = leg(node, order[index + 1])
        latest = min(float(tw_end[node]), latest - service[order[index + 1]] - travel)
    if latest < tw_start[order[0]]:
        return -1, -1, -1
    return occupancy, release, int(latest)


def vrptw_insert(route: np.ndarray, position: int, node: int) -> np.ndarray:

    return np.concatenate(
        (route[:position], np.asarray((node,), dtype=np.int64), route[position:])
    )


def vrptw_route_cost(
    route: np.ndarray,
    points: np.ndarray,
    demands: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    capacity: int,
) -> tuple[bool, float]:

    feasible, cost, _ = vrptw_route_metrics_seq(
        route, int(route.shape[0]), points, demands, tw_start, tw_end, service, capacity
    )
    return bool(feasible), float(cost)


def vrptw_eliminate_locally(
    routes: list[np.ndarray],
    target: int,
    points: np.ndarray,
    demands: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    capacity: int,
    max_elimination_rounds: int = vrptw_RESTORE_ROUNDS,
    max_drop_per_scope: int = vrptw_RESTORE_DROPS,
    max_sources_per_round: int = vrptw_RESTORE_SOURCES,
) -> tuple[list[np.ndarray], int, int]:

    from numba.typed import List

    if not routes:
        return [], 0, 0
    working = List([np.ascontiguousarray(route, dtype=np.int64).copy() for route in routes])
    removed, attempts = eliminate_routes(
        working, target, points, demands, tw_start, tw_end, service,
        capacity, max_elimination_rounds, max_drop_per_scope, max_sources_per_round,
    )
    return [route for route in working if len(route)], removed, attempts


@njit(cache=False)
def vrptw_two_opt_route(
    route: np.ndarray,
    coords: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    candidates: np.ndarray,
    passes: int,
) -> tuple[np.ndarray, int]:

    best = route.copy()
    length = best.shape[0]
    moves = 0
    if length < 4:
        return best, moves
    position = np.empty(coords.shape[0], dtype=np.int64)
    for _ in range(passes):
        improved = False
        for index in range(coords.shape[0]):
            position[index] = -1
        for index in range(length):
            position[best[index]] = index
        feasible, cost, _ = vrptw_simulate_route(best, coords, tw_start, tw_end, service)
        if not feasible:
            return best, moves
        for i in range(length - 2):
            for slot in range(candidates.shape[1]):
                node = candidates[best[i], slot]
                if node < 0:
                    continue
                j = position[node]
                if j < 0 or j >= length or best[j] != node:
                    continue
                if j <= i + 1 or (i == 0 and j == length - 1):
                    continue
                low = i + 1
                high = j
                while low < high:
                    temporary = best[low]
                    best[low] = best[high]
                    best[high] = temporary
                    low += 1
                    high -= 1
                new_feasible, new_cost, _ = vrptw_simulate_route(
                    best, coords, tw_start, tw_end, service
                )
                if new_feasible and new_cost < cost:
                    cost = new_cost
                    moves += 1
                    improved = True
                    for index in range(length):
                        position[best[index]] = index
                else:
                    low = i + 1
                    high = j
                    while low < high:
                        temporary = best[low]
                        best[low] = best[high]
                        best[high] = temporary
                        low += 1
                        high -= 1
        if not improved:
            break
    return best, moves


@njit(cache=False)
def vrptw_route_metrics_seq(
    route: np.ndarray,
    length: int,
    coords: np.ndarray,
    demands: np.ndarray,
    tw_start: np.ndarray,
    tw_end: np.ndarray,
    service: np.ndarray,
    capacity: int,
) -> tuple[bool, int, int]:

    load = 0
    cost = 0
    time = tw_start[0]
    previous = 0
    feasible = True
    for index in range(length):
        node = route[index]
        load += demands[node]
        travel = vrptw_edge(coords, previous, node)
        cost += travel
        time += travel
        if time < tw_start[node]:
            time = tw_start[node]
        if time > tw_end[node]:
            feasible = False
        time += service[node]
        previous = node
    travel = vrptw_edge(coords, previous, 0)
    cost += travel
    time += travel
    if time > tw_end[0]:
        feasible = False
    if load > capacity:
        feasible = False
    return feasible, cost, load


@njit(cache=True)
def eliminate_routes(routes, target, points, demands, starts, ends, service,
                     capacity, rounds, max_drops, max_sources):
    removed = 0
    attempts = 0
    for _ in range(rounds):
        active = 0
        for route in routes:
            active += int(len(route) > 0)
        if active <= target or removed >= max_drops:
            break
        lengths = np.array([len(route) if len(route) else len(points)+1 for route in routes])
        source_order = np.argsort(lengths, kind='mergesort')
        dropped = False
        for source in source_order[:max_sources]:
            candidate = routes[source]
            if not len(candidate):
                continue
            attempts += 1
            working = [route.copy() for route in routes]
            costs = np.zeros(len(routes), dtype=np.int64)
            loads = np.zeros(len(routes), dtype=np.int64)
            valid = True
            for index, route in enumerate(working):
                if not len(route):
                    continue
                feasible, cost, load = vrptw_route_metrics_seq(
                    route, len(route), points, demands, starts, ends, service, capacity)
                costs[index], loads[index] = cost, load
                valid = valid and feasible
            if not valid:
                continue
            old_cost = np.sum(costs)
            emptied = True
            scratch = np.empty(len(points), dtype=np.int64)
            for node in candidate:
                best_delta = np.iinfo(np.int64).max
                best_other, best_position = -1, -1
                best_length = len(points)+1
                for other, route in enumerate(working):
                    if other == source or not len(route) or loads[other]+demands[node] > capacity:
                        continue
                    for position in range(len(route)+1):
                        left = route[position-1] if position else 0
                        right = route[position] if position < len(route) else 0
                        delta = vrptw_edge(points,left,node)+vrptw_edge(points,node,right)-vrptw_edge(points,left,right)
                        if delta > best_delta or (delta == best_delta and len(route) >= best_length):
                            continue
                        for index in range(len(route)+1):
                            scratch[index] = node if index == position else route[index if index < position else index-1]
                        feasible, _, _ = vrptw_route_metrics_seq(
                            scratch,len(route)+1,points,demands,starts,ends,service,capacity)
                        if feasible:
                            best_delta, best_other, best_position = delta, other, position
                            best_length = len(route)
                if best_other < 0:
                    emptied = False
                    break
                route = working[best_other]
                updated = np.empty(len(route)+1,dtype=np.int64)
                for index in range(len(updated)):
                    updated[index] = node if index == best_position else route[index if index < best_position else index-1]
                working[best_other] = updated
                costs[best_other] += best_delta
                loads[best_other] += demands[node]
            # Fewer vehicles are useful only when they improve our distance objective.
            if emptied and np.sum(costs)-costs[source] < old_cost:
                working[source] = np.empty(0,dtype=np.int64)
                for index in range(len(routes)):
                    routes[index] = working[index]
                removed += 1
                dropped = True
                break
        if not dropped:
            break
    return removed, attempts


@njit(cache=True)
def route_pair_codes(routes, neighbors, node_count):
    owner = np.full(node_count,-1,dtype=np.int64)
    for ri, route in enumerate(routes):
        for node in route:
            owner[node] = ri
    codes = np.empty(node_count*neighbors.shape[1],dtype=np.int64)
    count = 0
    width = len(routes)
    for route in routes:
        for node in route:
            source = owner[node]
            for neighbor in neighbors[node]:
                if neighbor <= 0:
                    continue
                target = owner[neighbor]
                if target >= 0 and source != target:
                    codes[count] = min(source,target)*width+max(source,target)
                    count += 1
    return np.unique(codes[:count])


@njit(cache=True)
def improve_pair_batch(routes, pairs, begin, end, points, demands, starts, ends,
                       service, capacity, neighbors, max_moves):
    relocate = 0
    exchange = 0
    owner = np.full(len(points),-1,dtype=np.int64)
    positions = np.full(len(points),-1,dtype=np.int64)
    for ri,route in enumerate(routes):
        for position,node in enumerate(route):
            owner[node],positions[node] = ri,position
    longest = 0
    for route in routes:
        longest = max(longest,len(route))
    scratch_a = np.empty(2*longest+2,dtype=np.int64)
    scratch_b = np.empty(2*longest+2,dtype=np.int64)
    for index in range(begin,end):
        left,right = pairs[index,0],pairs[index,1]
        a,b = routes[left],routes[right]
        if not len(a) or not len(b):
            continue
        needed = len(a)+len(b)+2
        if needed > len(scratch_a):
            scratch_a = np.empty(needed,dtype=np.int64)
            scratch_b = np.empty(needed,dtype=np.int64)
        direction,position,insert,delta,node = best_relocate_fast(
            a,b,left,right,owner,positions,points,demands,starts,ends,service,
            capacity,neighbors,scratch_a,scratch_b)
        cut_a,cut_b,exchange_delta = best_tail_exchange_fast(
            a,b,points,demands,starts,ends,service,capacity)
        if delta < 0 and delta <= exchange_delta:
            source,target = (a,b) if direction == 0 else (b,a)
            new_source = np.empty(len(source)-1,dtype=np.int64)
            new_target = np.empty(len(target)+1,dtype=np.int64)
            for slot in range(len(new_source)):
                new_source[slot] = source[slot if slot < position else slot+1]
            for slot in range(len(new_target)):
                new_target[slot] = node if slot == insert else target[slot if slot < insert else slot-1]
            if direction == 0:
                routes[left],routes[right] = new_source,new_target
            else:
                routes[right],routes[left] = new_source,new_target
            relocate += 1
        elif exchange_delta < 0:
            routes[left] = np.concatenate((a[:cut_a+1],b[cut_b+1:]))
            routes[right] = np.concatenate((b[:cut_b+1],a[cut_a+1:]))
            exchange += 1
        else:
            continue
        for ri in (left,right):
            for position,node in enumerate(routes[ri]):
                owner[node],positions[node] = ri,position
        if relocate+exchange >= max_moves:
            break
    return relocate,exchange


@njit(cache=True)
def route_summaries(route,points,demands,starts,ends,service):
    length = len(route)
    cost = np.zeros(length+1,dtype=np.int64)
    load = np.zeros(length+1,dtype=np.int64)
    departure = np.empty(length+1,dtype=np.int64)
    latest = np.empty(length,dtype=np.int64)
    departure[0] = starts[0]
    previous = 0
    for index,node in enumerate(route):
        travel = vrptw_edge(points,previous,node)
        cost[index+1] = cost[index]+travel
        load[index+1] = load[index]+demands[node]
        departure[index+1] = max(departure[index]+travel,starts[node])+service[node]
        previous = node
    next_node = 0
    next_latest = ends[0]
    for index in range(length-1,-1,-1):
        node = route[index]
        latest[index] = min(ends[node],next_latest-service[node]-vrptw_edge(points,node,next_node))
        next_node,next_latest = node,latest[index]
    total = cost[length]+vrptw_edge(points,previous,0)
    return cost,load,departure,latest,total


@njit(cache=True)
def best_tail_exchange_fast(a,b,points,demands,starts,ends,service,capacity):
    ca,la,da,ta,cost_a = route_summaries(a,points,demands,starts,ends,service)
    cb,lb,db,tb,cost_b = route_summaries(b,points,demands,starts,ends,service)
    best_delta = 0
    best_a,best_b = -1,-1
    for ia in range(len(a)+1):
        end_a = a[ia-1] if ia else 0
        first_a = a[ia] if ia<len(a) else 0
        for ib in range(len(b)+1):
            if ia+len(b)-ib == 0 or ib+len(a)-ia == 0:
                continue
            if la[ia]+lb[-1]-lb[ib] > capacity or lb[ib]+la[-1]-la[ia] > capacity:
                continue
            end_b = b[ib-1] if ib else 0
            first_b = b[ib] if ib<len(b) else 0
            join_a = vrptw_edge(points,end_a,first_b)
            join_b = vrptw_edge(points,end_b,first_a)
            tail_a = cost_a-ca[ia+1] if ia<len(a) else 0
            tail_b = cost_b-cb[ib+1] if ib<len(b) else 0
            delta = ca[ia]+join_a+tail_b+cb[ib]+join_b+tail_a-cost_a-cost_b
            if delta >= best_delta:
                continue
            limit_a = tb[ib] if ib<len(b) else ends[0]
            limit_b = ta[ia] if ia<len(a) else ends[0]
            if da[ia]+join_a > limit_a or db[ib]+join_b > limit_b:
                continue
            best_delta,best_a,best_b = delta,ia-1,ib-1
    return best_a,best_b,best_delta


@njit(cache=True)
def best_relocate_fast(a,b,left,right,owner,positions,points,demands,
                       starts,ends,service,capacity,neighbors,scratch_a,scratch_b):
    _,cost_a,load_a = vrptw_route_metrics_seq(a,len(a),points,demands,starts,ends,service,capacity)
    _,cost_b,load_b = vrptw_route_metrics_seq(b,len(b),points,demands,starts,ends,service,capacity)
    best_delta = 0
    best_direction,best_position,best_insert,best_node = 0,0,0,-1
    for direction in range(2):
        source,target = (a,b) if direction == 0 else (b,a)
        target_id = right if direction == 0 else left
        target_load = load_b if direction == 0 else load_a
        if len(source)<2:
            continue
        eligible = np.zeros(len(target)+1,dtype=np.bool_)
        for position,node in enumerate(source):
            if target_load+demands[node] > capacity:
                continue
            eligible[:] = False
            for anchor in neighbors[node]:
                if anchor > 0 and owner[anchor] == target_id:
                    index = positions[anchor]
                    eligible[index] = True
                    eligible[index+1] = True
            if not np.any(eligible):
                continue
            left_node = source[position-1] if position else 0
            right_node = source[position+1] if position+1<len(source) else 0
            removal = vrptw_edge(points,left_node,right_node)-vrptw_edge(points,left_node,node)-vrptw_edge(points,node,right_node)
            valid_source = False
            checked_source = False
            for insert in range(len(target)+1):
                if not eligible[insert]:
                    continue
                previous = target[insert-1] if insert else 0
                following = target[insert] if insert<len(target) else 0
                delta = removal+vrptw_edge(points,previous,node)+vrptw_edge(points,node,following)-vrptw_edge(points,previous,following)
                if delta >= best_delta:
                    continue
                if not checked_source:
                    for slot in range(len(source)-1):
                        scratch_a[slot] = source[slot if slot < position else slot+1]
                    valid_source,_,_ = vrptw_route_metrics_seq(scratch_a,len(source)-1,points,demands,starts,ends,service,capacity)
                    checked_source = True
                if not valid_source:
                    break
                for slot in range(len(target)+1):
                    scratch_b[slot] = node if slot == insert else target[slot if slot < insert else slot-1]
                valid,_,_ = vrptw_route_metrics_seq(scratch_b,len(target)+1,points,demands,starts,ends,service,capacity)
                if valid:
                    best_delta,best_direction,best_position,best_insert,best_node = delta,direction,position,insert,node
    return best_direction,best_position,best_insert,best_delta,best_node


@njit(cache=True)
def pair_priorities(points,nodes,offsets,neighbors):
    count=len(offsets)-1
    owner=np.full(len(points),-1,dtype=np.int64)
    previous=np.zeros(len(points),dtype=np.int64)
    following=np.zeros(len(points),dtype=np.int64)
    for ri in range(count):
        begin,end=offsets[ri],offsets[ri+1]
        for index in range(begin,end):
            node=nodes[index]
            owner[node]=ri
            previous[node]=nodes[index-1] if index>begin else 0
            following[node]=nodes[index+1] if index+1<end else 0
    priorities=np.full((count,count),np.inf)
    for node in nodes:
        source=owner[node]
        saving=edge_distance(points,previous[node],node)+edge_distance(points,node,following[node])-edge_distance(points,previous[node],following[node])
        for anchor in neighbors[node]:
            target=owner[anchor]
            if anchor<=0 or target<0 or source==target:continue
            before=edge_distance(points,previous[anchor],node)+edge_distance(points,node,anchor)-edge_distance(points,previous[anchor],anchor)
            after=edge_distance(points,anchor,node)+edge_distance(points,node,following[anchor])-edge_distance(points,anchor,following[anchor])
            delta=min(before,after)-saving
            left,right=min(source,target),max(source,target)
            priorities[left,right]=min(priorities[left,right],delta)
    return priorities


def refine_cross_routes(problem,routes,neighbors,max_moves=EVALUATION_CROSS_MOVES):
    if max_moves<0:
        raise ValueError("cross-route move budget must be nonnegative")
    working=[route.copy() for route in routes]
    if max_moves==0 or len(working)<2:
        return working,0
    nodes=np.concatenate(working)
    offsets=np.concatenate(([0],np.cumsum([len(route) for route in working]))).astype(np.int64)
    priorities=pair_priorities(problem.points,nodes,offsets,neighbors)
    left,right=np.nonzero(np.isfinite(priorities))
    order=np.lexsort((right,left,priorities[left,right]))[:EVALUATION_ROUTE_PAIRS]
    # A pair can exchange all of its customers. Scratch space must remain safe
    # after earlier moves grow a route, not just for the initial maximum length.
    scratch_a=np.empty(len(nodes)+2,dtype=np.int64)
    scratch_b=np.empty(len(nodes)+2,dtype=np.int64)
    moves=0
    for index in order:
        ri,rj=int(left[index]),int(right[index])
        a,b=working[ri],working[rj]
        direction,position,insert_at,delta,node=vrptw_best_relocate(
            a,b,problem.points,problem.demands,problem.tw_start,problem.tw_end,
            problem.service_time,int(problem.capacity),neighbors,scratch_a,scratch_b)
        cut_a,cut_b,exchange=vrptw_best_tail_exchange(
            a,b,problem.points,problem.demands,problem.tw_start,problem.tw_end,
            problem.service_time,int(problem.capacity),scratch_a,scratch_b)
        if delta<0 and delta<=exchange:
            source,target=(a,b) if direction==0 else (b,a)
            if not (0<=position<len(source) and source[position]==node and 0<=insert_at<=len(target)):
                raise ValueError("invalid bounded relocate result")
            source=np.delete(source,position)
            target=np.insert(target,insert_at,node)
            working[ri],working[rj]=(source,target) if direction==0 else (target,source)
        elif exchange<0:
            working[ri]=np.concatenate((a[:cut_a+1],b[cut_b+1:]))
            working[rj]=np.concatenate((b[:cut_b+1],a[cut_a+1:]))
        else:continue
        moves+=1
        if moves>=max_moves:break
    return working,moves


def vrptw_optimize(
    problem: vrptw_VRPTWProblem,
    routes: list[np.ndarray],
    max_moves: int = vrptw_OPTIMIZE_MOVES,
    max_passes: int = vrptw_OPTIMIZE_PASSES,
    candidate_k: int = vrptw_OPTIMIZE_CANDIDATE_K,
    time_budget: float | None = None,
    candidate_matrix: np.ndarray | None = None,
) -> vrptw_OptimizeResult:
    from numba.typed import List
    from scipy.spatial import cKDTree

    started = time.perf_counter()
    budget = vrptw_OPTIMIZE_SECONDS if time_budget is None else max(0.0,time_budget)
    working = [np.ascontiguousarray(route,dtype=np.int64).copy() for route in routes if len(route)]
    if len(working) < 2 or budget <= 0 or max_moves <= 0:
        return vrptw_OptimizeResult(working, {"relocate":0,"tail_exchange":0,"identity":1})
    points = np.ascontiguousarray(problem.points)
    if candidate_matrix is None:
        _,nearest = cKDTree(points).query(points,k=min(candidate_k+1,len(points)),workers=1)
        candidate_matrix = np.asarray(nearest,dtype=np.int64).reshape(len(points),-1)
    working = List(working)
    codes = route_pair_codes(working,candidate_matrix,len(points))[:vrptw_OPTIMIZE_MAX_PAIRS]
    pairs = np.column_stack((codes//len(working),codes%len(working)))
    moves = {"relocate":0,"tail_exchange":0}
    # Check the deadline between small compiled batches, retaining responsive budgets
    # without a Python dispatch for every route pair and every accepted operation.
    for _ in range(max_passes):
        improved = 0
        for begin in range(0,len(pairs),128):
            remaining_moves = max_moves-sum(moves.values())
            if remaining_moves <= 0 or time.perf_counter()-started >= budget:
                break
            relocate,exchange = improve_pair_batch(
                working,pairs,begin,min(begin+128,len(pairs)),points,
                problem.demands,problem.tw_start,problem.tw_end,problem.service_time,
                int(problem.capacity),candidate_matrix,remaining_moves,
            )
            moves["relocate"] += relocate
            moves["tail_exchange"] += exchange
            improved += relocate+exchange
        if not improved or sum(moves.values()) >= max_moves or time.perf_counter()-started >= budget:
            break
    moves.update(identity=int(not sum(moves.values())),rejected_relocate=0,
                 rejected_tail_exchange=0,candidate_pairs=len(pairs),
                 seconds=time.perf_counter()-started)
    return vrptw_OptimizeResult([route for route in working if len(route)],moves)




def compress(problem: object, seed: int = 0) -> PreparedCompression:
    """Build compact paths while preserving the original recovery representation."""
    result = vrptw_compress(
        problem,
        builder=lambda p: vrptw_build_routes_lookahead(p, vrptw_BUILDER_CANDIDATE_K),
    )

    paths = [np.asarray(p, dtype=np.int64) for p in result.recovery["sequences"]]

    offsets = np.asarray(result.recovery["sub_offsets"])

    sources = [source for source in range(len(offsets) - 1)
               for _ in range(int(offsets[source]), int(offsets[source + 1]))]

    return PreparedCompression(result, paths, sources)


def optimize(problem: object, compression: object, routes: list[np.ndarray], seed: int = 0):
    """Refine restored routes with the original problem-specific operators."""
    deadline = time.perf_counter()+vrptw_OPTIMIZE_SECONDS
    recovery = compression.recovery

    decision = compression.decision

    points = np.ascontiguousarray(problem.points, dtype=np.float64)

    demands = np.ascontiguousarray(problem.demands, dtype=np.int64)

    starts = np.ascontiguousarray(problem.tw_start, dtype=np.int64)

    ends = np.ascontiguousarray(problem.tw_end, dtype=np.int64)

    service = np.ascontiguousarray(problem.service_time, dtype=np.int64)

    candidates = np.ascontiguousarray(recovery["candidates"], dtype=np.int64)

    by_source: list[list[np.ndarray]] = [
        [] for _ in range(int(decision["unit_count"]))
    ]

    node_source = np.asarray(recovery["node_unit"], dtype=np.int64)

    for route in routes:
        by_source[int(node_source[int(route[0])])].append(route)

    repaired: list[np.ndarray] = []

    for source, local_routes in enumerate(by_source):
        target = int(decision["unit_vehicle_lb"][source])
        if len(local_routes) > target and time.perf_counter()<deadline:
            local_routes, _, _ = vrptw_eliminate_locally(
                local_routes, target, points, demands, starts, ends, service,
                int(problem.capacity),
            )
        repaired.extend(local_routes)

    polished = []

    for route in repaired:
        if time.perf_counter() >= deadline:
            polished.append(route.copy())
            continue
        improved, _ = vrptw_two_opt_route(
            route, points, starts, ends, service, candidates,
            vrptw_RESTORE_POLISH_PASSES,
        )
        polished.append(np.ascontiguousarray(improved, dtype=np.int64))

    best = polished

    best_cost = route_set_cost(problem,best)

    from scipy.spatial import cKDTree
    _, nearest = cKDTree(points).query(
        points, k=min(vrptw_OPTIMIZE_CANDIDATE_K+1,len(points)), workers=1,
    )
    nearest = np.asarray(nearest,dtype=np.int64).reshape(len(points),-1)
    orders = vrptw_restarts(polished, VRPTW_STARTS)
    for index, order in enumerate(orders):
        remaining = deadline-time.perf_counter()
        if remaining<=0:
            break
        found = vrptw_optimize(
            problem, order, time_budget=remaining/(len(orders)-index),
            candidate_matrix=nearest,
        ).routes
        cost = route_set_cost(problem, found)
        if best_cost is None or cost < best_cost:
            best, best_cost = found, cost

    return best



def route_set_cost(problem, routes: list[np.ndarray]) -> int:

    points = problem.points
    total = 0
    for route in routes:
        previous = 0
        for node_value in route:
            node = int(node_value)
            total += int(math.floor(math.hypot(
                float(points[previous][0] - points[node][0]),
                float(points[previous][1] - points[node][1])) + 0.5))
            previous = node
        total += int(math.floor(math.hypot(
            float(points[previous][0] - points[0][0]),
            float(points[previous][1] - points[0][1])) + 0.5))
    return total


def vrptw_restarts(routes: list[np.ndarray], count: int) -> list[list[np.ndarray]]:

    if count <= 1:
        return [list(routes)]
    orders = [list(routes), list(reversed(routes)),
              sorted(routes, key=lambda route: (len(route), int(route[0])))]
    rng = np.random.default_rng(SEARCH_SEED)
    while len(orders) < count:
        order = rng.permutation(len(routes))
        orders.append([routes[int(index)] for index in order])
    return orders[:count]



VRPTWProblem = vrptw_VRPTWProblem
