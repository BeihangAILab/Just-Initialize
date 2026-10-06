"""Small, move-bounded downstream trajectories for initialization evaluation."""
from __future__ import annotations
import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from .recovery import edge_distance, feasible_route, get_recovery

EVALUATION_NEIGHBORS = 16
EVALUATION_MOVES = 8


@njit(cache=True)
def bounded_two_opt(points, nodes, offsets, candidates, depot, demands, capacity,
                    starts, ends, service, temporal, max_moves):
    positions = np.full(len(points), -1, dtype=np.int64)
    for index,node in enumerate(nodes):
        positions[node] = index
    accepted = 0
    for ri in range(len(offsets)-1):
        begin,end = offsets[ri],offsets[ri+1]
        for p in range(begin if depot < 0 else begin-1,end-1):
            a = nodes[p] if p>=begin else depot
            b = nodes[p+1]
            for c in candidates[a]:
                q = positions[c]
                if q < p+2 or q>=end:
                    continue
                d = nodes[q+1] if q+1<end else (nodes[begin] if depot<0 else depot)
                gain = edge_distance(points,a,b)+edge_distance(points,c,d)-edge_distance(points,a,c)-edge_distance(points,b,d)
                if gain <= 0:
                    continue
                lo,hi = p+1,q
                while lo<hi:
                    nodes[lo],nodes[hi]=nodes[hi],nodes[lo]
                    lo+=1;hi-=1
                valid = not temporal or feasible_route(points,nodes[begin:end],demands,capacity,starts,ends,service,True)
                if not valid:
                    lo,hi=p+1,q
                    while lo<hi:
                        nodes[lo],nodes[hi]=nodes[hi],nodes[lo]
                        lo+=1;hi-=1
                    continue
                for index in range(p+1,q+1):
                    positions[nodes[index]]=index
                accepted += 1
                break
            if accepted>=max_moves:
                return nodes
    return nodes


def evaluate_refinement(compact, routes):
    if not hasattr(compact,"_evaluation_neighbors"):
        points = np.asarray(compact.problem.points,dtype=np.float64)
        if compact.kind == "cvrp":
            points = np.vstack((points,compact.problem.depot))
            depot = len(points)-1
        else:
            depot = -1 if compact.kind == "tsp" else 0
        width = min(len(points),EVALUATION_NEIGHBORS+1)
        _,neighbors=cKDTree(points).query(points,k=width,workers=1)
        compact._evaluation_neighbors=(points,np.asarray(neighbors,dtype=np.int64).reshape(len(points),width),depot)
    points,neighbors,depot=compact._evaluation_neighbors
    nodes=np.concatenate(routes).astype(np.int64,copy=True)
    offsets=np.concatenate(([0],np.cumsum([len(route) for route in routes]))).astype(np.int64)
    recovery=get_recovery(compact)
    bounded_two_opt(points,nodes,offsets,neighbors,depot,recovery.demands,recovery.capacity,
                    recovery.starts,recovery.ends,recovery.service,compact.kind=="vrptw",EVALUATION_MOVES)
    result=[nodes[a:b].copy() for a,b in zip(offsets[:-1],offsets[1:])]
    if compact.kind == "vrptw":
        from ..problems.vrptw import refine_cross_routes
        result,moves=refine_cross_routes(compact.problem,result,neighbors)
        compact._evaluation_cross_moves=getattr(compact,"_evaluation_cross_moves",0)+moves
    if compact.kind == "tsp":
        return result[0]
    if compact.kind == "pctsp":
        return np.concatenate(([0],result[0],[0])).astype(np.int64)
    return result
