"""Feasibility and objective evaluation in compact space."""
from __future__ import annotations
import math
import hashlib
import numpy as np

from .representation import Compact, Unit, Item, Plan, ProblemSpec, problem_spec
from .recovery import get_recovery
from .representation import DOWNSTREAM_SAMPLES

def recovery_rng(compact: Compact, plan: Plan) -> np.random.Generator:
    # A plan-specific stream makes S=1 scores repeatable and cacheable without
    # changing the distribution according to candidate evaluation order.
    key = tuple(tuple((item.unit, item.reverse) for item in route) for route in plan.routes)
    digest = hashlib.blake2b(repr((compact.seed, key)).encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def unit_boundary(problem, unit: Unit, edge) -> tuple[tuple[int, int, int, bool], ...]:

    profiles = []
    for path in (unit.nodes, unit.nodes[::-1]):
        duration = 0
        ready = 0
        latest = None
        feasible = True
        previous = None
        for node_value in path:
            node = int(node_value)
            if previous is not None:
                travel = int(edge(previous, node))
                duration += travel
                ready += travel
            start = int(problem.tw_start[node])
            end = int(problem.tw_end[node])
            if ready < start:
                ready = start
            if ready > end:
                feasible = False
            limit = end - duration
            latest = limit if latest is None else min(latest, limit)
            service = int(problem.service_time[node])
            duration += service
            ready += service
            previous = node
        profiles.append((duration, ready, latest, feasible))
    return tuple(profiles)


class Oracle:

    def __init__(self, compact: Compact, spec: ProblemSpec | None = None):
        self.compact = compact
        self.recovery = get_recovery(compact)
        self.downstream_cache = {}
        self.downstream_evaluations = 0
        self.problem = compact.problem
        self.kind = compact.kind
        self.spec = problem_spec(compact.kind) if spec is None else spec
        self.points = np.asarray(compact.problem.points)
        self.depot = (np.asarray(compact.problem.depot) if self.kind == "cvrp"
                      else self.points[0] if self.kind != "tsp" else None)
        self.optional = self.spec.optional_nodes
        self.single_route = self.spec.single_route
        self.quota = int(compact.problem.quota) if self.spec.prize_quota else 0
        self.capacity = int(compact.problem.capacity) if self.spec.capacity_limited else 0
        self.internal = np.asarray([
            sum(self.edge(int(a), int(b)) for a, b in zip(unit.nodes[:-1], unit.nodes[1:]))
            for unit in compact.units
        ], dtype=np.int64)
        self.total_penalty = (int(np.sum(compact.problem.penalties[1:]))
                              if self.optional else 0)
        count = len(compact.units)

        self.profile = ({id(unit): unit_boundary(compact.problem, unit, self.edge)
                         for unit in compact.units} if self.kind == "vrptw" else {})

        if self.kind == "vrptw":

            self.candidate_width = min(count, 64)
            self.search_trials = min(200, max(40, count // 4))
            self.merge_budget = 0
        elif self.kind == "pctsp":
            self.candidate_width = min(count, 64)
            self.search_trials = min(64, max(16, count // 32))
            self.merge_budget = 0
        elif self.kind == "cvrp":
            self.candidate_width = min(count, 64)
            self.search_trials = min(4000, max(500, 25 * count))
            self.merge_budget = count
        else:
            self.candidate_width = min(count, 64)
            self.search_trials = min(400, max(80, count))
            self.merge_budget = 0

        self.route_points = np.asarray(self.points,dtype=np.float64)
        if self.kind=="cvrp":
            self.route_points=np.vstack((self.route_points,self.depot))
        self.depot_index = -1 if self.depot is None else (len(self.route_points)-1 if self.kind=="cvrp" else 0)
        self.endpoints=np.asarray([[unit.nodes[0],unit.nodes[-1]] for unit in compact.units],dtype=np.int64)
        self.unit_demands=np.asarray([unit.demand for unit in compact.units],dtype=np.int64)
        self.route_profiles=np.zeros((count,2,4),dtype=np.int64)
        if self.kind=="vrptw":
            for index,unit in enumerate(compact.units):
                self.route_profiles[index]=self.profile[id(unit)]

    def edge(self, a: int | None, b: int | None) -> int:
        pa = self.depot if a is None else self.points[a]
        pb = self.depot if b is None else self.points[b]
        return int(math.floor(math.hypot(float(pa[0] - pb[0]), float(pa[1] - pb[1])) + 0.5))

    def initial(self):
        return (None, 0, int(self.problem.tw_start[0]) if self.kind == "vrptw" else 0)

    def append(self, state, unit: Unit, reverse: bool):
        last, load, clock = state
        if self.capacity and load + unit.demand > self.capacity:
            return None
        path = unit.nodes[::-1] if reverse else unit.nodes
        first = int(path[0])
        end = int(path[-1])
        if last is None and self.depot is None:
            delta = 0
        else:
            delta = self.edge(last, first)
            if self.depot is not None:
                delta += self.edge(end, None) - self.edge(last, None) if last is not None else self.edge(end, None)
        if self.kind == "vrptw":
            duration, ready, latest, feasible = self.profile[id(unit)][1 if reverse else 0]
            arrival = clock + self.edge(last, first)
            if not feasible or latest is None or arrival > latest:
                return None
            clock = max(arrival + duration, ready)
            if clock + self.edge(end, None) > int(self.problem.tw_end[0]):
                return None
        return delta, (end, load + unit.demand, clock)

    def score(self, delta: int, unit: Unit, prize_so_far: int,
              state=None, reverse: bool = False) -> float:

        if self.optional:
            entry = int(unit.nodes[-1] if reverse else unit.nodes[0])
            return float(self.edge(state[0], entry))
        return float(delta)

    def select_units(self) -> set[int]:

        if not self.optional:
            return set(range(len(self.compact.units)))
        ranked = sorted(
            range(len(self.compact.units)),
            key=lambda index: (
                self.compact.units[index].penalty + 12 * self.compact.units[index].prize
            ) / max(1.0, float(np.linalg.norm(
                self.compact.units[index].center - self.depot
            ))),
            reverse=True,
        )
        selected: set[int] = set()
        prize = 0
        for index in ranked:
            selected.add(index)
            prize += self.compact.units[index].prize
            if prize >= self.quota:
                break
        if prize < self.quota:
            raise RuntimeError("compact units cannot meet the prize quota")
        return selected

    def route_feasible(self, items: list[Item]) -> bool:
        state = self.initial()
        for item in items:
            candidate = self.append(state, self.compact.units[item.unit], item.reverse)
            if candidate is None:
                return False
            state = candidate[1]
        return True

    def route_cost(self, items: list[Item]) -> float:
        from .geometry import compact_route_cost
        ids=np.fromiter((item.unit for item in items),dtype=np.int64,count=len(items))
        reverses=np.fromiter((item.reverse for item in items),dtype=np.bool_,count=len(items))
        return float(compact_route_cost(self.route_points,self.depot_index,ids,reverses,
                         self.endpoints,self.internal,self.unit_demands,self.capacity,self.route_profiles,
                         self.kind=="vrptw",int(self.problem.tw_start[0]) if self.kind=="vrptw" else 0,
                         int(self.problem.tw_end[0]) if self.kind=="vrptw" else 0))

    def objective(self, plan: Plan) -> float:
        value = sum(self.route_cost(route) for route in plan.routes)
        if not math.isfinite(value):
            return float("inf")
        if self.optional:
            selected = [item.unit for route in plan.routes for item in route]
            if sum(self.compact.units[index].prize for index in selected) < self.quota:
                return float("inf")
            value += self.total_penalty - sum(
                self.compact.units[index].penalty for index in selected
            )
        return value

    @staticmethod
    def plan_key(plan: Plan):
        return tuple(tuple((item.unit, item.reverse) for item in route) for route in plan.routes)

    def downstream_score(self, plan: Plan) -> float:
        from .pipeline import optimize
        key = self.plan_key(plan)
        if key not in self.downstream_cache:
            rng = recovery_rng(self.compact, plan)
            values = []
            for _ in range(DOWNSTREAM_SAMPLES):
                restored = self.recovery.sample(plan, rng).routes
                refined = optimize(self.compact, [route.copy() for route in restored], self.compact.seed, evaluation=True)
                values.append(self.solution_cost(refined))
            self.downstream_cache[key] = float(np.mean(values))
            self.downstream_evaluations += len(values)
        return self.downstream_cache[key]

    def solution_cost(self, solution) -> float:
        if self.kind == "pctsp":
            cycle = np.asarray(solution, dtype=np.int64)
            if len(cycle) < 3 or cycle[0] != 0 or cycle[-1] != 0:
                raise ValueError("PCTSP refinement returned an invalid depot cycle")
            routes = [cycle[1:-1]]
        elif self.kind == "tsp":
            routes = [np.asarray(solution, dtype=np.int64)]
        else:
            routes = [np.asarray(route, dtype=np.int64) for route in solution]
        nodes = np.concatenate(routes) if routes else np.empty(0, dtype=np.int64)
        required = set(range(len(self.points))) if self.kind in ("tsp", "cvrp") else set(range(1, len(self.points)))
        selected = set(int(node) for node in nodes)
        if len(nodes) != len(selected) or not selected <= required:
            raise ValueError("downstream refinement returned duplicate or unknown nodes")
        if not self.optional and selected != required:
            raise ValueError("downstream refinement did not cover all required nodes")
        if self.optional and int(self.problem.prizes[nodes].sum()) < self.quota:
            raise ValueError("downstream refinement violated the prize quota")
        cost = 0
        for route in routes:
            if not len(route) or not self.recovery._feasible(route):
                raise ValueError("downstream refinement returned an infeasible route")
            cost += sum(self.edge(int(a), int(b)) for a, b in zip(route[:-1], route[1:]))
            if self.depot is None:
                cost += self.edge(int(route[-1]), int(route[0]))
            else:
                cost += self.edge(None, int(route[0])) + self.edge(int(route[-1]), None)
        if self.optional:
            cost += int(sum(self.problem.penalties[node] for node in required - selected))
        return float(cost)
