"""CVRP representation, compression and local optimization."""
from __future__ import annotations

from ..core.representation import PreparedCompression
import numpy as np
from numba import njit
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from typing import Any

cvrp_CUSTOMER_NEIGHBORS = 32


cvrp_SAVINGS_LAMBDA = 1.4


cvrp_FRAGMENT_RHO = 0.8


cvrp_PORTAL_NEIGHBORS = 16


cvrp_POLISH_NEIGHBORS = 16


cvrp_POLISH_MOVES_PER_ROUTE = 10


cvrp_EXCHANGE_PREP_POLISH_MOVES = 1


cvrp_EXCHANGE_NEIGHBORS = 8


cvrp_EXCHANGE_MOVES = 4


@dataclass(frozen=True)
class cvrp_CVRPProblem:
    name: str
    points: np.ndarray
    demands: np.ndarray
    depot: np.ndarray
    capacity: int
    vehicle_limit: int | None
    edge_weight_type: str
    reference_value: int | None = None
    reference_type: str | None = None


@dataclass
class cvrp_CompressionResult:
    decision: Any
    recovery: Any
    original_size: int
    decision_size: int
    representation_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.original_size <= 1 or not 0 < self.decision_size < self.original_size:
            raise ValueError("compression must strictly reduce the decision dimension")
        if self.representation_bytes < 0:
            raise ValueError("representation_bytes must be non-negative")


@dataclass
class cvrp_OptimizeResult:
    routes: list[np.ndarray]
    moves: dict[str, int] = field(default_factory=dict)


def cvrp_rounded(values: np.ndarray) -> np.ndarray:

    return np.floor(values + 0.5).astype(np.int64)


@njit(cache=True)
def cvrp_link_savings(left,right,order,demands,capacity):
    n=len(demands)
    parent=np.arange(n)
    loads=demands.copy()
    heads=np.arange(n)
    tails=np.arange(n)
    following=np.full(n,-1,dtype=np.int64)
    preceding=np.full(n,-1,dtype=np.int64)
    active=np.ones(n,dtype=np.bool_)
    for edge in order:
        first,second=left[edge],right[edge]
        rf,rs=first,second
        while parent[rf]!=rf:
            parent[rf]=parent[parent[rf]]
            rf=parent[rf]
        while parent[rs]!=rs:
            parent[rs]=parent[parent[rs]]
            rs=parent[rs]
        if rf==rs or loads[rf]+loads[rs]>capacity:continue
        if first!=heads[rf] and first!=tails[rf]:continue
        if second!=heads[rs] and second!=tails[rs]:continue
        for root,reverse in ((rf,first==heads[rf]),(rs,second==tails[rs])):
            if reverse:
                node=heads[root]
                while node>=0:
                    next_node=following[node]
                    following[node],preceding[node]=preceding[node],following[node]
                    node=next_node
                heads[root],tails[root]=tails[root],heads[root]
        following[tails[rf]]=heads[rs]
        preceding[heads[rs]]=tails[rf]
        tails[rf]=tails[rs]
        parent[rs]=rf
        loads[rf]+=loads[rs]
        active[rs]=False
    nodes=np.empty(n,dtype=np.int64)
    offsets=np.empty(n+1,dtype=np.int64)
    count=0;position=0
    for root in range(n):
        if active[root]:
            offsets[count]=position
            node=heads[root]
            while node>=0:
                nodes[position]=node;position+=1
                node=following[node]
            count+=1
    offsets[count]=position
    return nodes,offsets[:count+1]


def cvrp_cw_routes(problem: cvrp_CVRPProblem, lam: float = cvrp_SAVINGS_LAMBDA) -> list[np.ndarray]:

    n = len(problem.points)
    k = min(cvrp_CUSTOMER_NEIGHBORS + 1, n)
    _, neighbor_rows = cKDTree(problem.points).query(problem.points, k=k, workers=1)
    neighbor_rows = np.asarray(neighbor_rows, dtype=np.int64)
    left = np.repeat(np.arange(n, dtype=np.int64), k - 1)
    right = neighbor_rows[:, 1:].reshape(-1)
    keep = left < right
    left, right = left[keep], right[keep]
    depot_cost = cvrp_rounded(
        np.hypot(
            problem.points[:, 0] - problem.depot[0],
            problem.points[:, 1] - problem.depot[1],
        )
    )
    pair_cost = cvrp_rounded(
        np.hypot(
            problem.points[left, 0] - problem.points[right, 0],
            problem.points[left, 1] - problem.points[right, 1],
        )
    )
    savings = depot_cost[left] + depot_cost[right] - lam * pair_cost
    order = np.argsort(-savings, kind="stable")
    nodes,offsets=cvrp_link_savings(left,right,order,np.asarray(problem.demands,dtype=np.int64),problem.capacity)
    return [nodes[a:b].copy() for a,b in zip(offsets[:-1],offsets[1:])]



def cvrp_slice_routes(
    problem: cvrp_CVRPProblem, routes: list[np.ndarray], limit: int
) -> list[np.ndarray]:

    fragments: list[np.ndarray] = []
    for route in routes:
        cur: list[int] = []
        load = 0
        for node in route:
            d = int(problem.demands[node])
            if cur and load + d > limit:
                fragments.append(np.asarray(cur, dtype=np.int64))
                cur, load = [], 0
            cur.append(int(node))
            load += d
        if cur:
            fragments.append(np.asarray(cur, dtype=np.int64))
    return fragments


def cvrp_fragment_features(points: np.ndarray, fragments: list[np.ndarray]):

    m = len(fragments)
    portals = np.asarray(
        [[points[frag[0]], points[frag[-1]]] for frag in fragments], dtype=float
    )
    fragment_sizes = np.asarray([len(frag) for frag in fragments], dtype=np.int64)
    from ..core.geometry import portal_pair_costs
    pair_cost = portal_pair_costs(portals)
    if m <= 1:
        neighbors = np.empty((m, 0), dtype=np.int64)
    else:
        degree = min(cvrp_PORTAL_NEIGHBORS, m - 1)
        np.fill_diagonal(pair_cost, np.inf)
        neighbors = np.argpartition(pair_cost, degree - 1, axis=1)[:, :degree].astype(
            np.int64
        )
    return portals, fragment_sizes, pair_cost, neighbors


def cvrp_optimize_polish(
    problem: cvrp_CVRPProblem,
    routes: list[np.ndarray],
    max_moves: int = cvrp_POLISH_MOVES_PER_ROUTE,
) -> cvrp_OptimizeResult:

    pts = problem.points.astype(np.float64)
    n = len(pts)
    _, global_nn = cKDTree(pts).query(pts, k=min(cvrp_POLISH_NEIGHBORS + 1, n), workers=1)
    global_nn = np.asarray(global_nn[:, 1:], dtype=np.int64)
    pos = np.full(n, -1, dtype=np.int64)
    output, total_moves = [], 0
    for route in routes:
        pos[:] = -1
        improved, moves = cvrp_two_opt_vec(
            route.copy().astype(np.int64), pts, global_nn, pos, max_moves
        )
        output.append(improved)
        total_moves += moves
    return cvrp_OptimizeResult(
        output,
        {
            "candidate_two_opt": total_moves,
            "candidate_neighbors": cvrp_POLISH_NEIGHBORS,
            "moves_per_route": max_moves,
            "vectorized": True,
        },
    )


def cvrp_two_opt_vec(
    route: np.ndarray,
    pts: np.ndarray,
    global_nn: np.ndarray,
    pos: np.ndarray,
    max_moves: int,
) -> tuple[np.ndarray, int]:

    count = len(route)
    if count < 4:
        return route, 0
    moves = 0
    while moves < max_moves:
        pos[route] = np.arange(count)
        a = np.arange(count - 1)
        b = a + 1
        ca = route[a]
        cand = pos[global_nn[ca]]
        valid = (cand > b[:, None]) & (cand < count - 1)
        flat_a = np.repeat(a, global_nn.shape[1])[valid.ravel()]
        flat_c = cand[valid]
        if len(flat_c) == 0:
            break
        flat_b = flat_a + 1
        flat_d = flat_c + 1
        ca_ = route[flat_a]
        cb_ = route[flat_b]
        cc_ = route[flat_c]
        cd_ = route[flat_d]
        p = pts
        dab = np.hypot(p[ca_, 0] - p[cb_, 0], p[ca_, 1] - p[cb_, 1])
        dac = np.hypot(p[ca_, 0] - p[cc_, 0], p[ca_, 1] - p[cc_, 1])
        dbd = np.hypot(p[cb_, 0] - p[cd_, 0], p[cb_, 1] - p[cd_, 1])
        dcd = np.hypot(p[cc_, 0] - p[cd_, 0], p[cc_, 1] - p[cd_, 1])
        delta = dac + dbd - dab - dcd
        best = int(np.argmin(delta))
        if delta[best] >= 0:
            break
        fb = int(flat_b[best])
        fc = int(flat_c[best])
        route[fb : fc + 1] = route[fb : fc + 1][::-1]
        moves += 1
    return route, moves


def cvrp_compress(problem: cvrp_CVRPProblem) -> cvrp_CompressionResult:

    routes = cvrp_cw_routes(problem, cvrp_SAVINGS_LAMBDA)
    routes = [
        np.asarray(x, dtype=np.int64) for x in cvrp_optimize_polish(problem, routes).routes
    ]
    limit = max(int(problem.demands.max()), int(problem.capacity * cvrp_FRAGMENT_RHO))
    fragments = cvrp_slice_routes(problem, routes, limit)
    centers = np.asarray([problem.points[f].mean(axis=0) for f in fragments])
    demands = np.asarray(
        [int(problem.demands[f].sum()) for f in fragments], dtype=np.int64
    )
    portals, fragment_sizes, pair_cost, neighbors = cvrp_fragment_features(
        problem.points, fragments
    )
    decision = {
        "unit_count": len(fragments),
        "centers": centers,
        "unit_demands": demands,
        "portals": portals,
        "fragment_sizes": fragment_sizes,
        "neighbors": neighbors,
        "pair_cost": pair_cost,
        "fragment_rho": cvrp_FRAGMENT_RHO,
    }
    representation_bytes = int(
        centers.nbytes
        + demands.nbytes
        + portals.nbytes
        + fragment_sizes.nbytes
        + neighbors.nbytes
        + pair_cost.nbytes
    )
    metadata = {
        "family": "path.v3",
        "units": len(fragments),
        "fragment_rho": cvrp_FRAGMENT_RHO,
        "cw_routes_c1": len(routes),
        "lambda": cvrp_SAVINGS_LAMBDA,
        "polish": "vectorized_2opt_10",
        "portal_neighbors": int(neighbors.shape[1] if neighbors.size else 0),
    }
    return cvrp_CompressionResult(
        decision,
        {"fragments": fragments},
        len(problem.points),
        len(fragments),
        representation_bytes,
        metadata,
    )


def cvrp_distance(p: np.ndarray, q: np.ndarray) -> int:

    return int(np.floor(np.hypot(p[0] - q[0], p[1] - q[1]) + 0.5))


def cvrp_best_exchange(problem: cvrp_CVRPProblem, routes: list[np.ndarray], nearest: np.ndarray):

    n = len(problem.points)
    route_of = np.full(n, -1, dtype=np.int64)
    position = np.full(n, -1, dtype=np.int64)
    loads = [0] * len(routes)
    prefixes = [None] * len(routes)
    for rid, route in enumerate(routes):
        if len(route) == 0:
            continue
        route_of[route] = rid
        position[route] = np.arange(len(route), dtype=np.int64)
        prefix = np.cumsum(problem.demands[route], dtype=np.int64)
        prefixes[rid] = prefix
        loads[rid] = int(prefix[-1])
    best, selected = 0, None
    for first in range(n):
        ra = int(route_of[first])
        if ra < 0:
            continue
        pa = int(position[first])
        route_a = routes[ra]
        a_next = (
            problem.depot if pa + 1 == len(route_a) else problem.points[route_a[pa + 1]]
        )
        tail_a = loads[ra] - int(prefixes[ra][pa])
        for second in nearest[first, 1:]:
            second = int(second)
            rb, pb = int(route_of[second]), int(position[second])
            if ra == rb:
                continue
            route_b = routes[rb]
            tail_b = loads[rb] - int(prefixes[rb][pb])
            if (
                loads[ra] - tail_a + tail_b > problem.capacity
                or loads[rb] - tail_b + tail_a > problem.capacity
            ):
                continue
            b_next = (
                problem.depot
                if pb + 1 == len(route_b)
                else problem.points[route_b[pb + 1]]
            )
            delta = (
                cvrp_distance(problem.points[first], b_next)
                + cvrp_distance(problem.points[second], a_next)
                - cvrp_distance(problem.points[first], a_next)
                - cvrp_distance(problem.points[second], b_next)
            )
            if delta < best:
                best, selected = delta, (ra, pa, rb, pb)
    return selected


def cvrp_optimize_exchange(problem: cvrp_CVRPProblem, routes: list[np.ndarray]) -> cvrp_OptimizeResult:

    ordered = cvrp_optimize_polish(problem, routes, cvrp_EXCHANGE_PREP_POLISH_MOVES)
    output = [route.copy() for route in ordered.routes]
    _, nearest = cKDTree(problem.points).query(
        problem.points, k=min(cvrp_EXCHANGE_NEIGHBORS + 1, len(problem.points)), workers=1
    )
    nearest = np.asarray(nearest, dtype=np.int64)
    moves = 0
    while moves < cvrp_EXCHANGE_MOVES:
        selected = cvrp_best_exchange(problem, output, nearest)
        if selected is None:
            break
        ra, pa, rb, pb = selected
        left, right = output[ra], output[rb]
        output[ra] = np.concatenate([left[: pa + 1], right[pb + 1 :]])
        output[rb] = np.concatenate([right[: pb + 1], left[pa + 1 :]])
        output = [route for route in output if len(route) > 0]
        moves += 1
    return cvrp_OptimizeResult(
        output,
        {
            "candidate_two_opt": ordered.moves.get("candidate_two_opt", 0),
            "candidate_two_optstar": moves,
            "candidate_neighbors": cvrp_EXCHANGE_NEIGHBORS,
        },
    )



def compress(problem: object, seed: int = 0) -> PreparedCompression:
    """Build compact paths while preserving the original recovery representation."""
    result = cvrp_compress(problem)

    paths = [np.asarray(p, dtype=np.int64) for p in result.recovery["fragments"]]

    sources = list(range(len(paths)))

    return PreparedCompression(result, paths, sources)


def optimize(problem: object, compression: object, routes: list[np.ndarray], seed: int = 0):
    """Refine restored routes with the original problem-specific operators."""
    return cvrp_optimize_exchange(problem, routes).routes



CVRPProblem = cvrp_CVRPProblem
