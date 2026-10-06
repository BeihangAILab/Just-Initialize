"""Compress, solve, restore and optimize a routing instance."""
from __future__ import annotations
import heapq
import numpy as np
from scipy.spatial import cKDTree

from .representation import Compact, Unit, Item, Plan, SEARCH_NEIGHBORS, MERGE_NEIGHBORS, DOWNSTREAM_SHORTLIST, DOWNSTREAM_SEARCH_ROUNDS
from .oracle import Oracle
from ..problems import get_adapter

def compress(kind: str, problem: object, seed: int = 0) -> Compact:
    prepared = get_adapter(kind).compress(problem, seed)
    units = []
    for path, source in zip(prepared.paths, prepared.sources, strict=True):
        points = problem.points[path]
        demand = int(problem.demands[path].sum()) if kind in ("cvrp", "vrptw") else 0
        prize = int(problem.prizes[path].sum()) if kind == "pctsp" else 0
        penalty = int(problem.penalties[path].sum()) if kind == "pctsp" else 0
        units.append(Unit(path, points.mean(axis=0), demand, prize, penalty, source))
    return Compact(kind, problem, units, prepared.source, seed)

def solve(compact: Compact, oracle: Oracle) -> Plan:

    units = compact.units
    centers = np.asarray([unit.center for unit in units])
    tree = cKDTree(centers)
    candidates = [_construct_plan(compact, oracle, tree, centers)]
    if oracle.merge_budget:
        candidates.append(_merge_plan(oracle, tree))
    initial = min(candidates, key=oracle.downstream_score)
    return _improve_plan(initial, oracle, tree)


def _construct_plan(compact: Compact, oracle: Oracle, tree: cKDTree,
                    centers: np.ndarray) -> Plan:
    units = compact.units
    remaining = oracle.select_units()
    routes: list[list[Item]] = []
    current: list[Item] = []
    state = oracle.initial()
    prize = 0
    while remaining:
        if current:
            query = oracle.points[state[0]]
        elif oracle.depot is not None:
            query = oracle.depot
        else:
            query = centers[min(remaining)]
        width = oracle.candidate_width
        requested = width
        candidates = []
        while not candidates:
            nearby = np.atleast_1d(tree.query(query, k=requested)[1]).tolist()
            candidates = [int(i) for i in nearby if int(i) in remaining][:width]
            if not candidates and requested < len(units):
                requested = min(len(units), requested * 2)
            elif not candidates:
                raise RuntimeError("unvisited compact unit is missing from spatial index")
        best = None
        for index in candidates:
            unit = units[index]
            for reverse in (False, True) if len(unit.nodes) > 1 else (False,):
                appended = oracle.append(state, unit, reverse)
                if appended is None:
                    continue
                delta, next_state = appended
                score = oracle.score(delta, unit, prize, state, reverse)
                if best is None or score < best[0]:
                    best = (score, index, reverse, next_state)
        if best is None:
            if not current or oracle.single_route:
                raise RuntimeError("compact search cannot place a required unit")
            routes.append(current)
            current = []
            state = oracle.initial()
            continue
        _, index, reverse, state = best
        current.append(Item(index, reverse))
        remaining.remove(index)
        prize += units[index].prize
    if current:
        routes.append(current)
    if oracle.optional and prize < oracle.quota:
        raise RuntimeError("compact search failed to collect the required prize")
    if not routes:
        raise RuntimeError("compact search produced no route")
    return Plan(routes)


def _merge_plan(oracle: Oracle, tree: cKDTree) -> Plan:

    units = oracle.compact.units
    active = {index: [Item(index)] for index in range(len(units))}
    costs = {index: oracle.route_cost(route) for index, route in active.items()}
    versions = {index: 0 for index in active}
    loads = {index: unit.demand for index,unit in enumerate(units)}
    heap = []

    def reversed_route(route: list[Item]) -> list[Item]:
        return [Item(item.unit, not item.reverse) for item in reversed(route)]

    def offer(left: int, right: int) -> None:
        if left == right or left not in active or right not in active:
            return
        if loads[left]+loads[right]>oracle.capacity:
            return
        best_gain = 0.0
        best_route = None
        for a in (active[left], reversed_route(active[left])):
            for b in (active[right], reversed_route(active[right])):
                joined = a + b
                last_item,first_item=a[-1],b[0]
                last_unit,first_unit=units[last_item.unit],units[first_item.unit]
                last=int(last_unit.nodes[0] if last_item.reverse else last_unit.nodes[-1])
                first=int(first_unit.nodes[-1] if first_item.reverse else first_unit.nodes[0])
                cost=costs[left]+costs[right]-oracle.edge(last,None)-oracle.edge(None,first)+oracle.edge(last,first)
                gain = costs[left] + costs[right] - cost
                if gain > best_gain:
                    best_gain, best_route = gain, joined
        if best_route is not None:
            heapq.heappush(heap, (-best_gain, left, right, versions[left],
                                  versions[right], best_route))

    width = min(len(units), MERGE_NEIGHBORS)
    for index, unit in enumerate(units):
        near = np.atleast_1d(tree.query(unit.center, k=width)[1]).tolist()
        for other in near:
            if int(other) > index:
                offer(index, int(other))
    merges = 0
    while heap and merges < oracle.merge_budget:
        negative_gain, left, right, left_version, right_version, route = heapq.heappop(heap)
        if (left not in active or right not in active or
                versions[left] != left_version or versions[right] != right_version):
            continue
        active[left] = route
        costs[left] = costs[left] + costs[right] + negative_gain
        loads[left]+=loads[right]
        del active[right], costs[right],loads[right]
        versions[left] += 1
        merges += 1
        for other in list(active):
            if other != left:
                offer(left, other)
                offer(other, left)
    return Plan(list(active.values()))


def _improve_plan(plan: Plan, oracle: Oracle, tree: cKDTree) -> Plan:
    units = oracle.compact.units
    rng = np.random.default_rng(oracle.compact.seed)
    best_value = oracle.downstream_score(plan)
    rounds = min(2 if oracle.compact.kind=="tsp" else DOWNSTREAM_SEARCH_ROUNDS, oracle.search_trials)
    for batch in np.array_split(np.arange(oracle.search_trials), rounds):
        current = plan.routes
        shortlist = {}
        for trial in batch:
            source_route = int(rng.integers(len(current)))
            if not current[source_route]:
                continue
            source_pos = int(rng.integers(len(current[source_route])))
            source_item = current[source_route][source_pos]
            near = np.atleast_1d(tree.query(
                units[source_item.unit].center, k=min(len(units), SEARCH_NEIGHBORS)
            )[1]).tolist()
            target_unit = int(near[int(rng.integers(len(near)))])
            locations = {item.unit: (ri, pi) for ri, route in enumerate(current)
                         for pi, item in enumerate(route)}
            candidate = [route.copy() for route in current]
            mode = trial % 4
            if target_unit not in locations:
                if not oracle.optional:
                    continue
                ri, pi = source_route, source_pos
                if mode == 0:
                    candidate[ri][pi] = Item(target_unit, False)
                else:
                    candidate[ri].insert(pi + 1, Item(target_unit, False))
            else:
                target_route, target_pos = locations[target_unit]
                if mode == 0:
                    candidate[source_route][source_pos], candidate[target_route][target_pos] = (
                        candidate[target_route][target_pos], candidate[source_route][source_pos]
                    )
                elif mode == 1:
                    item = candidate[source_route].pop(source_pos)
                    if source_route == target_route and source_pos < target_pos:
                        target_pos -= 1
                    candidate[target_route].insert(target_pos, item)
                elif mode == 2 and source_route == target_route:
                    left, right = sorted((source_pos, target_pos))
                    candidate[source_route][left:right + 1] = [
                        Item(item.unit, not item.reverse)
                        for item in reversed(candidate[source_route][left:right + 1])
                    ]
                elif oracle.optional:
                    candidate[source_route].pop(source_pos)
                else:
                    continue
            candidate = [route for route in candidate if route]
            trial_plan = Plan(candidate)
            value = oracle.objective(trial_plan)
            if not np.isfinite(value):
                continue
            key = oracle.plan_key(trial_plan)
            if key == oracle.plan_key(plan):
                continue
            shortlist[key] = (value, trial_plan)
            if len(shortlist) > DOWNSTREAM_SHORTLIST:
                worst = max(shortlist, key=lambda key: shortlist[key][0])
                del shortlist[worst]
        for _, candidate in sorted(shortlist.values(), key=lambda entry: entry[0]):
            value = oracle.downstream_score(candidate)
            if value < best_value:
                plan, best_value = candidate, value
    return plan


def restore(plan: Plan, compact: Compact) -> list[np.ndarray]:
    from .recovery import get_recovery
    from .oracle import recovery_rng
    return get_recovery(compact).sample(plan, recovery_rng(compact, plan)).routes


def optimize(compact: Compact, routes: list[np.ndarray], seed: int = 0, *, evaluation: bool = False):
    if evaluation:
        from .refinement import evaluate_refinement
        return evaluate_refinement(compact, routes)
    return get_adapter(compact.kind).optimize(compact.problem, compact.source, routes, seed)

