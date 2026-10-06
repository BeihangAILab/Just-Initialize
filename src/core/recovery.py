"""Geometry and sequence consistency guided recovery from compact plans.

Candidates keep every selected unit contiguous and preserve the compact unit order.
Only local path boundary choices vary. Symmetric routing edges are undirected in
E(q); reversing a path therefore preserves its local adjacencies.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.distance import pdist
from numba import njit
from .representation import Compact, Plan

RECOVERY_CANDIDATES = 4
RECOVERY_TEMPERATURE = 0.01
EPSILON = 1e-12


@njit(cache=True)
def edge_distance(points, a, b):
    return int(math.floor(math.hypot(points[a, 0]-points[b, 0], points[a, 1]-points[b, 1])+0.5))


@njit(cache=True)
def boundary_choices(points, reference, diameter, previous, following):
    length = len(reference)
    values = np.empty((length * (2 if length > 1 else 1), 3))
    count = 0
    for shift in range(length):
        for reverse in range(2 if length > 1 else 1):
            first, last = reference[shift], reference[(shift-1)%length]
            if reverse:
                first,last=last,first
            geometry = 0.0
            pairs = 0
            if previous >= 0:
                connection = edge_distance(points, previous, reference[0])
                geometry += (edge_distance(points,first,reference[0])/(diameter+EPSILON)
                             +abs(edge_distance(points,previous,first)-connection)/(connection+EPSILON))/3
                pairs += 1
            if following >= 0:
                connection = edge_distance(points,reference[-1],following)
                geometry += (edge_distance(points,last,reference[-1])/(diameter+EPSILON)
                             +abs(edge_distance(points,last,following)-connection)/(connection+EPSILON))/3
                pairs += 1
            sequence = 1.0/max(1,length-1) if shift else (1.0 if length==1 else 0.0)
            values[count,0]=0.5*((geometry/pairs if pairs else 0.0)+sequence)
            values[count,1]=shift
            values[count,2]=reverse
            count += 1
    return values


@njit(cache=True)
def sequence_discrepancy(path, positions):
    shared=0
    for index in range(len(path)-1):
        if abs(positions[path[index]]-positions[path[index+1]]) == 1:
            shared += 1
    return 1.0-shared/max(1,len(path)-1)


@njit(cache=True)
def feasible_route(points, route, demands, capacity, starts, ends, service, temporal):
    if capacity and np.sum(demands[route]) > capacity:
        return False
    if temporal:
        clock=starts[0]
        previous=0
        for node in route:
            clock=max(clock+edge_distance(points,previous,node),starts[node])
            if clock>ends[node]:
                return False
            clock += service[node]
            previous=node
        if clock+edge_distance(points,previous,0)>ends[0]:
            return False
    return True


def get_recovery(compact):
    if not hasattr(compact, "_recovery"):
        compact._recovery = Recovery(compact)
    return compact._recovery


@dataclass
class RecoveryCandidate:
    routes: list[np.ndarray]
    score: float
    geometry_loss: float
    sequence_loss: float


class Recovery:
    def __init__(self, compact: Compact):
        self.compact = compact
        self.problem = compact.problem
        self.points = np.asarray(self.problem.points)
        self.choice_cache = {}
        self.positions = np.full(len(self.points), -1, dtype=np.int64)
        for unit in compact.units:
            self.positions[unit.nodes] = np.arange(len(unit.nodes))
        empty = np.empty(0, dtype=np.int64)
        self.demands = np.asarray(getattr(self.problem,"demands",empty),dtype=np.int64)
        self.starts = np.asarray(getattr(self.problem,"tw_start",empty),dtype=np.int64)
        self.ends = np.asarray(getattr(self.problem,"tw_end",empty),dtype=np.int64)
        self.service = np.asarray(getattr(self.problem,"service_time",empty),dtype=np.int64)
        self.capacity = int(getattr(self.problem,"capacity",0))
        self.diameters = []
        for unit in compact.units:
            diameter = float(np.max(pdist(self.points[unit.nodes]))) if len(unit.nodes) > 1 else 0.0
            self.diameters.append(float(math.floor(diameter + 0.5)))

    def distance(self, a: int, b: int) -> int:
        return int(edge_distance(self.points,a,b))

    def pair_loss(self, left, right, exit_node: int, entry_node: int) -> float:
        a = self.compact.units[left.unit].nodes
        b = self.compact.units[right.unit].nodes
        reference_exit = int(a[0] if left.reverse else a[-1])
        reference_entry = int(b[-1] if right.reverse else b[0])
        connection = self.distance(reference_exit, reference_entry)
        return (self.distance(exit_node, reference_exit) / (self.diameters[left.unit] + EPSILON)
                + self.distance(entry_node, reference_entry) / (self.diameters[right.unit] + EPSILON)
                + abs(self.distance(exit_node, entry_node) - connection) / (connection + EPSILON)) / 3.0

    def _choices(self, route, index):
        item = route[index]
        path = self.compact.units[item.unit].nodes
        reference = path[::-1] if item.reverse else path
        length = len(reference)
        cyclic = self.compact.kind == "tsp"
        previous = route[index - 1] if index or cyclic else None
        following = route[(index + 1) % len(route)] if index + 1 < len(route) or cyclic else None
        previous_node = -1
        following_node = -1
        if previous is not None:
            nodes = self.compact.units[previous.unit].nodes
            previous_node = int(nodes[0] if previous.reverse else nodes[-1])
        if following is not None:
            nodes = self.compact.units[following.unit].nodes
            following_node = int(nodes[-1] if following.reverse else nodes[0])
        key = (item.unit,item.reverse,previous_node,following_node)
        if key not in self.choice_cache:
            values = boundary_choices(self.points,reference,self.diameters[item.unit],previous_node,following_node)
            choices = [(float(row[0]),int(row[1]),bool(row[2])) for row in values]
            choices.sort(key=lambda value:(value[0],value[1],value[2]))
            self.choice_cache[key] = choices[:RECOVERY_CANDIDATES]
        return reference,self.choice_cache[key]

    def _candidate(self, plan: Plan, paths) -> RecoveryCandidate | None:
        geometry = []
        sequences = []
        routes = []
        for items, local in zip(plan.routes, paths, strict=True):
            for item, path in zip(items, local, strict=True):
                reference = self.compact.units[item.unit].nodes
                if item.reverse:
                    reference = reference[::-1]
                sequences.append(float(sequence_discrepancy(path,self.positions)))
            for index in range(len(items) - 1):
                geometry.append(self.pair_loss(items[index], items[index + 1],
                                               int(local[index][-1]), int(local[index + 1][0])))
            if self.compact.kind == "tsp":
                geometry.append(self.pair_loss(items[-1], items[0], int(local[-1][-1]), int(local[0][0])))
            route = np.concatenate(local).astype(np.int64, copy=False)
            if not self._feasible(route):
                return None
            routes.append(route)
        geo = float(np.mean(geometry)) if geometry else 0.0
        seq = float(np.mean(sequences)) if sequences else 0.0
        return RecoveryCandidate(routes, -0.5 * (geo + seq), geo, seq)

    def _feasible(self, route: np.ndarray) -> bool:
        return bool(feasible_route(self.points,route,self.demands,self.capacity,
                                   self.starts,self.ends,self.service,self.compact.kind=="vrptw"))

    def candidates(self, plan: Plan, rng: np.random.Generator) -> list[RecoveryCandidate]:
        originals = [[self.compact.units[item.unit].nodes[::-1] if item.reverse
                      else self.compact.units[item.unit].nodes for item in route] for route in plan.routes]
        baseline = self._candidate(plan, originals)
        if baseline is None:
            raise ValueError("compact plan has no feasible stored-sequence recovery")
        candidates = [baseline]
        signatures = {tuple(tuple(int(node) for node in route) for route in baseline.routes)}
        alternatives = []
        for ri, route in enumerate(plan.routes):
            for pi in range(len(route)):
                reference, choices = self._choices(route, pi)
                for loss, shift, reverse in choices:
                    if shift == 0 and not reverse:
                        continue
                    alternatives.append((loss, ri, pi, shift, reverse, reference))
        alternatives.sort(key=lambda entry: entry[:5])
        # Build distinct, low-discrepancy complete candidates before applying
        # Boltzmann sampling. Sampling at both levels can collapse the set to
        # copies of the reference path, especially at low temperature.
        pool = alternatives[:24]
        if pool:
            indices = rng.permutation(len(pool))
            # Always include the preferred non-reference boundary pair first.
            indices = [0] + [int(index) for index in indices if index != 0]
            for index in indices:
                _, ri, pi, shift, reverse, reference = pool[index]
                paths = [local.copy() for local in originals]
                path = np.roll(reference, -shift)
                paths[ri][pi] = path[::-1] if reverse else path
                candidate = self._candidate(plan, paths)
                if candidate is None:
                    continue
                signature = tuple(tuple(int(node) for node in route) for route in candidate.routes)
                if signature not in signatures:
                    signatures.add(signature)
                    candidates.append(candidate)
                if len(candidates) >= RECOVERY_CANDIDATES:
                    break
        return candidates

    def sample(self, plan: Plan, rng: np.random.Generator) -> RecoveryCandidate:
        candidates = self.candidates(plan, rng)
        scores = np.asarray([candidate.score for candidate in candidates])
        weights = np.exp((scores - scores.max()) / RECOVERY_TEMPERATURE)
        return candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
