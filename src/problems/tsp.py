"""TSP representation, compression and local optimization."""
from __future__ import annotations

from ..core.representation import PreparedCompression
import math
import numpy as np
import time
from dataclasses import dataclass, field
from numba import njit
from scipy.spatial import cKDTree
from typing import Any

tsp_CANDIDATE_K = 64


tsp_TARGET_FRAGMENT = 32


tsp_FRAGMENT_K = 8


tsp_PORTAL_NEIGHBORS = 16


tsp_LS_K = 64


tsp_LS_BASINS = 16

tsp_FINAL_REFINE_SECONDS = 2.0


tsp_LS_TOP_M = 80


tsp_LS_THREE_OPT_PASSES = 16


tsp_FULL_TWO_OPT_PASSES = 20


tsp_FULL_DONTLOOK_PASSES = 40


tsp_RESEARCH_TWO_OPT_PASSES = 10


tsp_RESEARCH_DONTLOOK_PASSES = 20


tsp_RESEARCH_THREE_OPT_PASSES = 4


tsp_RESEARCH_LK_DEPTH = 3


tsp_RESEARCH_LK_PASSES = 1


tsp_OR_OPT_SEGMENTS = (1, 2, 3)


tsp_OR_OPT_PASSES = 3


@dataclass(frozen=True)
class tsp_TSPProblem:
    name: str
    points: np.ndarray
    edge_weight_type: str
    bks: int | None = None
    group: str = ""


@dataclass
class tsp_CompressionResult:
    decision: Any
    recovery: Any
    original_size: int
    decision_size: int
    representation_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class tsp_BlueprintContext:
    seed: int
    deadline: float

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.perf_counter())

    def ensure_time(self, stage: str) -> None:
        if time.perf_counter() > self.deadline:
            raise TimeoutError(f"blueprint exceeded deadline during {stage}")


def tsp_c_build_candidates(points: np.ndarray, k: int) -> np.ndarray:

    count = len(points)
    if count == 0 or k == 0:
        return np.empty((count, k), dtype=np.int64)
    candidates = np.broadcast_to(np.arange(count, dtype=np.int64)[:, None],
                                 (count, k)).copy()
    if count == 1:
        return candidates

    take = min(k, count - 1)
    neighbors = np.asarray(
        cKDTree(points).query(points, k=take + 1, workers=-1)[1], dtype=np.int64
    )
    self_mask = neighbors == np.arange(count)[:, None]

    self_column = np.where(self_mask.any(axis=1), self_mask.argmax(axis=1), take + 1)
    columns = np.arange(take)[None, :]
    source_columns = columns + (columns >= self_column[:, None])
    candidates[:, :take] = np.take_along_axis(neighbors, source_columns, axis=1)
    return candidates


def tsp_c_pair_cost_matrix(portals: np.ndarray) -> np.ndarray:

    from ..core.geometry import portal_pair_costs
    return portal_pair_costs(portals)



@njit(cache=False)
def tsp_c_build_path_fragments(
    points: np.ndarray, candidates: np.ndarray, k: int, target: int
) -> tuple[np.ndarray, np.ndarray]:

    n = len(points)
    used = np.zeros(n, dtype=np.uint8)
    nodes = np.empty(n, dtype=np.int64)
    offsets = np.empty(n + 1, dtype=np.int64)
    output = 0
    fragment_count = 0
    scratch = np.empty(2 * target + 1, dtype=np.int64)
    midpoint = target
    for start in range(n):
        if used[start]:
            continue
        offsets[fragment_count] = output
        left = midpoint
        right = midpoint + 1
        scratch[midpoint] = start
        used[start] = 1
        length = 1
        while length < target:
            best = -1
            best_d2 = 1e300
            prepend = False
            for endpoint_side in range(2):
                current = scratch[left] if endpoint_side == 0 else scratch[right - 1]
                for ci in range(k):
                    nxt = candidates[current, ci]
                    if nxt == current or used[nxt]:
                        continue
                    dx = points[current, 0] - points[nxt, 0]
                    dy = points[current, 1] - points[nxt, 1]
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best = nxt
                        prepend = endpoint_side == 0
            if best < 0:
                break
            if prepend:
                left -= 1
                scratch[left] = best
            else:
                scratch[right] = best
                right += 1
            used[best] = 1
            length += 1
        for position in range(left, right):
            nodes[output] = scratch[position]
            output += 1
        fragment_count += 1
    offsets[fragment_count] = output
    return (nodes[:output], offsets[: fragment_count + 1])


def tsp_c_fragment_portals(
    points: np.ndarray, local_nodes: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    count = len(offsets) - 1
    portals = np.empty((count, 2, 2), dtype=np.float64)
    sizes = np.empty(count, dtype=np.int64)
    for fragment in range(count):
        start = int(offsets[fragment])
        end = int(offsets[fragment + 1])
        portals[fragment, 0] = points[local_nodes[start]]
        portals[fragment, 1] = points[local_nodes[end - 1]]
        sizes[fragment] = end - start
    return (portals, sizes)


def tsp_c_portal_neighbors(portals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    count = len(portals)
    if count <= 1:
        return (
            np.empty((count, 0), dtype=np.int64),
            np.zeros((count, count), dtype=np.float64),
        )
    degree = min(tsp_PORTAL_NEIGHBORS, count - 1)
    costs = tsp_c_pair_cost_matrix(portals)
    np.fill_diagonal(costs, np.inf)
    selected = np.argpartition(costs, degree - 1, axis=1)[:, :degree]
    selected_costs = np.take_along_axis(costs, selected, axis=1)
    order = np.argsort(selected_costs, axis=1, kind="stable")
    return (
        np.take_along_axis(selected, order, axis=1).astype(np.int64, copy=False),
        costs,
    )


def tsp_c_compress(problem: tsp_TSPProblem, context: tsp_BlueprintContext) -> tsp_CompressionResult:

    candidates = tsp_c_build_candidates(problem.points, tsp_CANDIDATE_K)
    local_nodes, offsets = tsp_c_build_path_fragments(
        problem.points, candidates, tsp_FRAGMENT_K, tsp_TARGET_FRAGMENT
    )
    portals, sizes = tsp_c_fragment_portals(problem.points, local_nodes, offsets)
    neighbors, pair_cost = tsp_c_portal_neighbors(portals)
    decision = {
        "portals": portals,
        "fragment_sizes": sizes,
        "neighbors": neighbors,
        "pair_cost": pair_cost,
    }
    recovery = {
        "local_nodes": local_nodes,
        "offsets": offsets,
        "candidates": candidates,
    }
    representation_bytes = sum(value.nbytes for value in decision.values()) + sum(
        value.nbytes for value in recovery.values()
    )
    return tsp_CompressionResult(
        decision=decision,
        recovery=recovery,
        original_size=len(problem.points),
        decision_size=len(portals),
        representation_bytes=representation_bytes,
        metadata={
            "method": "greedy_path_fragment_feature_supernodes_cached_portal_cost",
            "target_fragment": tsp_TARGET_FRAGMENT,
            "fragment_graph": "endpoint_portal_compatibility",
            "portal_cost_storage": "decision_cached_dense_matrix",
            "decision_features": [
                "endpoint_portals",
                "fragment_size",
                "portal_adjacency",
            ],
            "solve_uses": ["endpoint_portals", "fragment_size", "portal_adjacency"],
            "candidate_k": tsp_CANDIDATE_K,
            "three_opt": "full_4_types",

            "ils": {
                "basins": tsp_LS_BASINS,
                "k": tsp_LS_K,
                "top_m": tsp_LS_TOP_M,
                "three_opt_passes": tsp_LS_THREE_OPT_PASSES,
            },
        },
    )


@njit(cache=False)
def tsp_ils__relocate(
    tour: np.ndarray, pos: np.ndarray, p: int, seglen: int, q: int
) -> None:

    if q < p:
        seg0 = tour[p]
        seg1 = tour[p + 1] if seglen > 1 else seg0
        seg2 = tour[p + 2] if seglen > 2 else seg0
        for i in range(p - 1, q, -1):
            tour[i + seglen] = tour[i]
            pos[tour[i + seglen]] = i + seglen
        tour[q + 1] = seg0
        pos[seg0] = q + 1
        if seglen > 1:
            tour[q + 2] = seg1
            pos[seg1] = q + 2
        if seglen > 2:
            tour[q + 3] = seg2
            pos[seg2] = q + 3
    else:
        seg0 = tour[p]
        seg1 = tour[p + 1] if seglen > 1 else seg0
        seg2 = tour[p + 2] if seglen > 2 else seg0
        for i in range(p + seglen, q + 1):
            tour[i - seglen] = tour[i]
            pos[tour[i - seglen]] = i - seglen
        tour[q - seglen + 1] = seg0
        pos[seg0] = q - seglen + 1
        if seglen > 1:
            tour[q - seglen + 2] = seg1
            pos[seg1] = q - seglen + 2
        if seglen > 2:
            tour[q - seglen + 3] = seg2
            pos[seg2] = q - seglen + 3


@njit(cache=False)
def tsp_ils_or_opt(
    tour: np.ndarray,
    pos: np.ndarray,
    points: np.ndarray,
    candidates: np.ndarray,
    k: int,
    seglen: int,
    max_passes: int,
) -> None:

    count = len(tour)
    for _ in range(max_passes):
        improved = False
        for p in range(count):
            if p + seglen >= count:
                continue
            s0 = tour[p]
            slast = tour[p + seglen - 1]
            before = tour[(p - 1 + count) % count]
            after = tour[(p + seglen) % count]
            dx1 = points[before, 0] - points[s0, 0]
            dy1 = points[before, 1] - points[s0, 1]
            c1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            dx2 = points[slast, 0] - points[after, 0]
            dy2 = points[slast, 1] - points[after, 1]
            c2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            dx3 = points[before, 0] - points[after, 0]
            dy3 = points[before, 1] - points[after, 1]
            c3 = math.sqrt(dx3 * dx3 + dy3 * dy3)
            base_gain = c1 + c2 - c3
            best_gain = 0.0
            best_q = -1
            for cidx in range(k):
                q = candidates[before, cidx]
                if q == before:
                    continue
                qi = pos[q]
                if p <= qi < p + seglen:
                    continue
                if qi == (p - 1 + count) % count:
                    continue
                qn = tour[(qi + 1) % count]
                dx4 = points[q, 0] - points[s0, 0]
                dy4 = points[q, 1] - points[s0, 1]
                c4 = math.sqrt(dx4 * dx4 + dy4 * dy4)
                dx5 = points[slast, 0] - points[qn, 0]
                dy5 = points[slast, 1] - points[qn, 1]
                c5 = math.sqrt(dx5 * dx5 + dy5 * dy5)
                dx6 = points[q, 0] - points[qn, 0]
                dy6 = points[q, 1] - points[qn, 1]
                c6 = math.sqrt(dx6 * dx6 + dy6 * dy6)
                gain = base_gain + c6 - c4 - c5
                if gain > best_gain:
                    best_gain = gain
                    best_q = qi
            if best_q >= 0:
                tsp_ils__relocate(tour, pos, p, seglen, best_q)
                improved = True
        if not improved:
            break


@njit(cache=False)
def tsp_ils_two_opt_pruned(
    tour: np.ndarray,
    pos: np.ndarray,
    points: np.ndarray,
    candidates: np.ndarray,
    k: int,
    max_passes: int,
) -> None:

    count = len(tour)
    for _ in range(max_passes):
        improved = False
        for p in range(count):
            a = tour[p]
            b = tour[(p + 1) % count]
            ax = points[a, 0]
            ay = points[a, 1]
            dx_ab = ax - points[b, 0]
            dy_ab = ay - points[b, 1]
            cost_ab = math.sqrt(dx_ab * dx_ab + dy_ab * dy_ab)
            best_gain = 0.0
            best_q = -1
            for cidx in range(k):
                c = candidates[a, cidx]
                if c == a:
                    continue
                dx_ac = ax - points[c, 0]
                dy_ac = ay - points[c, 1]
                cost_ac = math.sqrt(dx_ac * dx_ac + dy_ac * dy_ac)
                if cost_ac >= cost_ab:
                    break
                q = pos[c]
                if q == p or q == (p + 1) % count or q == (p - 1 + count) % count:
                    continue
                d = tour[(q + 1) % count]
                dx_cd = points[c, 0] - points[d, 0]
                dy_cd = points[c, 1] - points[d, 1]
                cost_cd = math.sqrt(dx_cd * dx_cd + dy_cd * dy_cd)
                dx_bd = points[b, 0] - points[d, 0]
                dy_bd = points[b, 1] - points[d, 1]
                cost_bd = math.sqrt(dx_bd * dx_bd + dy_bd * dy_bd)
                gain = cost_ab + cost_cd - (cost_ac + cost_bd)
                if gain > best_gain:
                    best_gain = gain
                    best_q = q
            if best_q > 0:
                lo = (p + 1) % count
                hi = best_q
                length = (hi - lo + count) % count + 1
                steps = length // 2
                x = lo
                y = hi
                for _ in range(steps):
                    tour[x], tour[y] = (tour[y], tour[x])
                    pos[tour[x]] = x
                    pos[tour[y]] = y
                    x = (x + 1) % count
                    y = (y - 1 + count) % count
                improved = True
        if not improved:
            break


@njit(cache=False)
def tsp_ils_two_opt_dontlook_pruned(
    tour: np.ndarray,
    pos: np.ndarray,
    points: np.ndarray,
    candidates: np.ndarray,
    k: int,
    max_passes: int,
) -> None:

    count = len(tour)
    dontlook = np.zeros(count, dtype=np.uint8)
    for _ in range(max_passes):
        improved = False
        for p in range(count):
            a = tour[p]
            if dontlook[a]:
                continue
            b = tour[(p + 1) % count]
            ax = points[a, 0]
            ay = points[a, 1]
            dx_ab = ax - points[b, 0]
            dy_ab = ay - points[b, 1]
            cost_ab = math.sqrt(dx_ab * dx_ab + dy_ab * dy_ab)
            best_gain = 0.0
            best_q = -1
            best_c = -1
            for cidx in range(k):
                c = candidates[a, cidx]
                if c == a:
                    continue
                dx_ac = ax - points[c, 0]
                dy_ac = ay - points[c, 1]
                cost_ac = math.sqrt(dx_ac * dx_ac + dy_ac * dy_ac)
                if cost_ac >= cost_ab:
                    break
                q = pos[c]
                if q == p or q == (p + 1) % count or q == (p - 1 + count) % count:
                    continue
                d = tour[(q + 1) % count]
                dx_cd = points[c, 0] - points[d, 0]
                dy_cd = points[c, 1] - points[d, 1]
                cost_cd = math.sqrt(dx_cd * dx_cd + dy_cd * dy_cd)
                dx_bd = points[b, 0] - points[d, 0]
                dy_bd = points[b, 1] - points[d, 1]
                cost_bd = math.sqrt(dx_bd * dx_bd + dy_bd * dy_bd)
                gain = cost_ab + cost_cd - (cost_ac + cost_bd)
                if gain > best_gain:
                    best_gain = gain
                    best_q = q
                    best_c = c
            if best_q > 0:
                d = tour[(best_q + 1) % count]
                lo = (p + 1) % count
                hi = best_q
                length = (hi - lo + count) % count + 1
                steps = length // 2
                x = lo
                y = hi
                for _ in range(steps):
                    tour[x], tour[y] = (tour[y], tour[x])
                    pos[tour[x]] = x
                    pos[tour[y]] = y
                    x = (x + 1) % count
                    y = (y - 1 + count) % count
                dontlook[a] = 0
                dontlook[b] = 0
                dontlook[best_c] = 0
                dontlook[d] = 0
                improved = True
            else:
                dontlook[a] = 1
        if not improved:
            break


@njit(cache=False)
def tsp_ils__d2(points: np.ndarray, i: int, j: int) -> float:

    dx = points[i, 0] - points[j, 0]
    dy = points[i, 1] - points[j, 1]
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=False)
def tsp_ils_double_bridge(
    tour: np.ndarray, pos: np.ndarray, i: int, j: int, k: int, last: int
) -> None:

    n = len(tour)
    result = np.concatenate(
        (
            tour[0 : i + 1],
            tour[k + 1 : last + 1],
            tour[j + 1 : k + 1],
            tour[i + 1 : j + 1],
            tour[last + 1 : n],
        )
    )
    for idx in range(n):
        tour[idx] = result[idx]
        pos[result[idx]] = idx


@njit(cache=False)
def tsp_ils__tour_cost(points: np.ndarray, tour: np.ndarray) -> int:

    total = 0.0
    n = len(tour)
    for idx in range(n):
        a = tour[idx]
        b = tour[(idx + 1) % n]
        dx = points[a, 0] - points[b, 0]
        dy = points[a, 1] - points[b, 1]
        total += float(int(math.sqrt(dx * dx + dy * dy) + 0.5))
    return int(total)


@njit(cache=False)
def tsp_ils__lk_reverse(tour: np.ndarray, pos: np.ndarray, lo: int, hi: int) -> None:

    x = lo
    y = hi
    while True:
        tour[x], tour[y] = (tour[y], tour[x])
        pos[tour[x]] = x
        pos[tour[y]] = y
        if y - x <= 1:
            break
        x += 1
        y -= 1


@njit(cache=False)
def tsp_ils_lk_backtracking(
    tour: np.ndarray,
    pos: np.ndarray,
    points: np.ndarray,
    candidates: np.ndarray,
    k: int,
    max_depth: int,
    max_passes: int,
) -> None:

    n = len(tour)
    max_len = 2 * (max_depth + 1) + 2
    seq = np.empty(max_len, dtype=np.int64)
    best_seq = np.empty(max_len, dtype=np.int64)
    used = np.zeros(n, dtype=np.uint8)
    src_stack = np.empty(max_depth + 2, dtype=np.int64)
    seg_stack = np.empty(max_depth + 2, dtype=np.int64)
    gain_stack = np.empty(max_depth + 2, dtype=np.float64)
    ci_stack = np.empty(max_depth + 2, dtype=np.int64)
    base_stack = np.empty(max_depth + 2, dtype=np.float64)
    for _ in range(max_passes):
        improved = False
        for p in range(n):
            anchor = tour[p]
            seg_start = tour[(p + 1) % n]
            used[anchor] = 1
            used[seg_start] = 1
            seq[0] = seg_start
            depth = 0
            src_stack[0] = anchor
            seg_stack[0] = seg_start
            gain_stack[0] = tsp_ils__d2(points, anchor, seg_start)
            ci_stack[0] = 0
            base_stack[0] = gain_stack[0]
            best_gain = 0.0
            best_len = 0
            while depth >= 0:
                src = src_stack[depth]
                seg = seg_stack[depth]
                gain = gain_stack[depth]
                base = base_stack[depth]
                ci = ci_stack[depth]
                extended = False
                while ci < k:
                    dst = candidates[src, ci]
                    ci += 1
                    ci_stack[depth] = ci
                    if dst == src or used[dst]:
                        continue
                    if tsp_ils__d2(points, src, dst) >= base:
                        break
                    q = pos[dst]
                    if q <= pos[seg]:
                        continue
                    removed = tour[(q + 1) % n]
                    if used[removed]:
                        continue
                    new_gain = (
                        gain
                        - tsp_ils__d2(points, src, dst)
                        + tsp_ils__d2(points, dst, removed)
                    )
                    if new_gain <= 0.0:
                        continue
                    close_gain = new_gain - tsp_ils__d2(points, seg, removed)
                    if close_gain > best_gain:
                        best_gain = close_gain
                        best_len = 2 * depth + 2
                        for i in range(2 * depth + 1):
                            best_seq[i] = seq[i]
                        best_seq[2 * depth + 1] = dst
                    if depth < max_depth:
                        seq[2 * depth + 1] = dst
                        seq[2 * depth + 2] = removed
                        used[dst] = 1
                        used[removed] = 1
                        depth += 1
                        src_stack[depth] = seg
                        seg_stack[depth] = removed
                        gain_stack[depth] = new_gain
                        ci_stack[depth] = 0
                        base_stack[depth] = tsp_ils__d2(points, seg, removed)
                        extended = True
                        break
                if not extended:
                    if depth == 0:
                        break
                    used[seq[2 * depth - 1]] = 0
                    used[seq[2 * depth]] = 0
                    depth -= 1
            used[anchor] = 0
            used[seg_start] = 0
            if best_gain > 1e-09:
                blen = best_len
                for i in range(0, blen - 1, 2):
                    lo = pos[best_seq[i]]
                    hi = pos[best_seq[i + 1]]
                    if lo < hi:
                        tsp_ils__lk_reverse(tour, pos, lo, hi)
                improved = True
        if not improved:
            break


@njit(cache=False)
def tsp_ils__reverse_circ(
    tour: np.ndarray, pos: np.ndarray, lo: int, hi: int, n: int
) -> None:

    length = (hi - lo) % n + 1
    steps = length // 2
    for s in range(steps):
        x = (lo + s) % n
        y = (hi - s) % n
        tour[x], tour[y] = (tour[y], tour[x])
        pos[tour[x]] = x
        pos[tour[y]] = y


@njit(cache=False)
def tsp_ils_three_opt_full(
    tour: np.ndarray,
    pos: np.ndarray,
    points: np.ndarray,
    candidates: np.ndarray,
    k: int,
    max_passes: int,
) -> None:

    count = len(tour)
    for _ in range(max_passes):
        improved = False
        for p in range(count):
            a = tour[p]
            b = tour[(p + 1) % count]
            ab = tsp_ils__d2(points, a, b)
            best_gain = 0.0
            best_c = -1
            best_e = -1
            best_type = 0
            for ci in range(k):
                c = candidates[a, ci]
                if c == a or c == b:
                    continue
                ac = tsp_ils__d2(points, a, c)
                if ac >= ab:
                    break
                qc = pos[c]
                if qc <= p + 1:
                    continue
                d = tour[(qc + 1) % count]
                if d == a or d == b:
                    continue
                cd = tsp_ils__d2(points, c, d)
                bd = tsp_ils__d2(points, b, d)
                for ei in range(k):
                    e = candidates[b, ei]
                    if e == a or e == b or e == c or (e == d):
                        continue
                    be = tsp_ils__d2(points, b, e)
                    if be >= bd:
                        break
                    qe = pos[e]
                    if qe <= qc + 1:
                        continue
                    f = tour[(qe + 1) % count]
                    if f == a or f == b or f == c or (f == d) or (f == e):
                        continue
                    ef = tsp_ils__d2(points, e, f)
                    base = ab + cd + ef
                    df = tsp_ils__d2(points, d, f)
                    ae = tsp_ils__d2(points, a, e)
                    cf = tsp_ils__d2(points, c, f)
                    ad = tsp_ils__d2(points, a, d)
                    bf = tsp_ils__d2(points, b, f)
                    ce = tsp_ils__d2(points, c, e)
                    g4 = base - (ac + be + df)
                    g5 = base - (ae + bd + cf)
                    g6 = base - (ad + bf + ce)
                    g7 = base - (ad + be + cf)
                    if g4 > best_gain:
                        best_gain = g4
                        best_c = c
                        best_e = e
                        best_type = 4
                    if g5 > best_gain:
                        best_gain = g5
                        best_c = c
                        best_e = e
                        best_type = 5
                    if g6 > best_gain:
                        best_gain = g6
                        best_c = c
                        best_e = e
                        best_type = 6
                    if g7 > best_gain:
                        best_gain = g7
                        best_c = c
                        best_e = e
                        best_type = 7
            if best_gain > 0:
                qc = pos[best_c]
                qe = pos[best_e]
                if best_type == 4:
                    tsp_ils__reverse_circ(tour, pos, p + 1, qc, count)
                    tsp_ils__reverse_circ(tour, pos, qc + 1, qe, count)
                elif best_type == 5:
                    tsp_ils__reverse_circ(tour, pos, p + 1, qc, count)
                    tsp_ils__reverse_circ(tour, pos, qe + 1, p, count)
                elif best_type == 6:
                    tsp_ils__reverse_circ(tour, pos, qc + 1, qe, count)
                    tsp_ils__reverse_circ(tour, pos, qe + 1, p, count)
                else:
                    tsp_ils__reverse_circ(tour, pos, p + 1, qc, count)
                    tsp_ils__reverse_circ(tour, pos, qc + 1, qe, count)
                    tsp_ils__reverse_circ(tour, pos, qe + 1, p, count)
                improved = True
        if not improved:
            break


def tsp_ils__full_search(
    tour: np.ndarray,
    pos: np.ndarray,
    points: np.ndarray,
    candidates: np.ndarray,
    k: int,
    three_opt: int,
) -> None:

    tsp_ils_two_opt_pruned(tour, pos, points, candidates, k, tsp_FULL_TWO_OPT_PASSES)
    tsp_ils_two_opt_dontlook_pruned(
        tour, pos, points, candidates, k, tsp_FULL_DONTLOOK_PASSES
    )
    tsp_ils_three_opt_full(tour, pos, points, candidates, k, three_opt)
    for segment_length in tsp_OR_OPT_SEGMENTS:
        tsp_ils_or_opt(tour, pos, points, candidates, k, segment_length, tsp_OR_OPT_PASSES)
    tsp_ils_two_opt_pruned(tour, pos, points, candidates, k, tsp_FULL_TWO_OPT_PASSES)


def tsp_ils__research(
    tour: np.ndarray,
    pos: np.ndarray,
    points: np.ndarray,
    candidates: np.ndarray,
    k: int,
) -> None:

    tsp_ils_two_opt_pruned(tour, pos, points, candidates, k, tsp_RESEARCH_TWO_OPT_PASSES)
    tsp_ils_two_opt_dontlook_pruned(
        tour, pos, points, candidates, k, tsp_RESEARCH_DONTLOOK_PASSES
    )
    tsp_ils_three_opt_full(
        tour, pos, points, candidates, k, tsp_RESEARCH_THREE_OPT_PASSES
    )
    tsp_ils_lk_backtracking(
        tour, pos, points, candidates, k, tsp_RESEARCH_LK_DEPTH, tsp_RESEARCH_LK_PASSES
    )
    for segment_length in tsp_OR_OPT_SEGMENTS:
        tsp_ils_or_opt(tour, pos, points, candidates, k, segment_length, tsp_OR_OPT_PASSES)
    tsp_ils_two_opt_pruned(tour, pos, points, candidates, k, tsp_RESEARCH_TWO_OPT_PASSES)


def tsp_ils__longedge_kick(points, tour, pos, rng, top_m):

    n = len(tour)
    if n < 8:
        return
    d = points[tour] - points[np.roll(tour, -1)]
    edge_d2 = (d * d).sum(axis=1)
    longest = np.argsort(edge_d2)[::-1][: min(top_m, n)]
    if len(longest) < 4:
        return
    i, j, k, last = np.sort(rng.choice(longest, 4, replace=False))
    tsp_ils_double_bridge(tour, pos, int(i), int(j), int(k), int(last))


def tsp_ils_ils_optimize(
    problem,
    compressed,
    tour,
    context,
    basins=tsp_LS_BASINS,
    top_m=tsp_LS_TOP_M,
    three_opt=tsp_LS_THREE_OPT_PASSES,
):

    points = np.asarray(problem.points)
    candidates = compressed.recovery.get("candidates")
    result = np.asarray(tour, dtype=np.int64).copy()
    n = len(result)
    if candidates is None or n < 4:
        return result
    k = min(tsp_LS_K, int(candidates.shape[1]))
    pos = np.empty(n, dtype=np.int64)
    pos[result] = np.arange(n)
    tsp_ils__full_search(result, pos, points, candidates, k, three_opt)
    best = result.copy()
    best_cost = tsp_ils__tour_cost(points, best)
    rng = np.random.default_rng(context.seed)
    for _ in range(basins):
        if time.perf_counter()>=context.deadline:
            break
        trial = best.copy()
        tpos = np.empty(n, dtype=np.int64)
        tpos[trial] = np.arange(n)
        tsp_ils__longedge_kick(points, trial, tpos, rng, top_m)
        tsp_ils__research(trial, tpos, points, candidates, k)
        cost = tsp_ils__tour_cost(points, trial)
        if cost < best_cost:
            best_cost = cost
            best = trial.copy()
    return best



def compress(problem: object, seed: int = 0) -> PreparedCompression:
    """Build compact paths while preserving the original recovery representation."""
    context = tsp_BlueprintContext(seed, float("inf"))

    result = tsp_c_compress(problem, context)

    nodes, offsets = result.recovery["local_nodes"], result.recovery["offsets"]

    paths = [np.asarray(nodes[int(a):int(b)], dtype=np.int64)
             for a, b in zip(offsets[:-1], offsets[1:])]

    sources = list(range(len(paths)))

    return PreparedCompression(result, paths, sources)


def optimize(problem: object, compression: object, routes: list[np.ndarray], seed: int = 0):
    """Refine restored routes with the original problem-specific operators."""
    context = tsp_BlueprintContext(seed, time.perf_counter()+tsp_FINAL_REFINE_SECONDS)

    return tsp_ils_ils_optimize(problem, compression, routes[0], context)



TSPProblem = tsp_TSPProblem
