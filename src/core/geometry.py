"""Compiled geometry and compact route scoring kernels."""
import math
import numpy as np
from numba import njit
from .recovery import edge_distance


@njit(cache=True)
def portal_pair_costs(portals):
    count=len(portals)
    costs=np.empty((count,count),dtype=np.float64)
    for i in range(count):
        for j in range(count):
            best=np.inf
            for a in range(2):
                for b in range(2):
                    dx=portals[i,a,0]-portals[j,b,0]
                    dy=portals[i,a,1]-portals[j,b,1]
                    best=min(best,dx*dx+dy*dy)
            costs[i,j]=math.sqrt(best)
    return costs


@njit(cache=True)
def compact_route_cost(points,depot,ids,reverses,endpoints,internal,demands,
                       capacity,profiles,temporal,initial_clock,depot_end):
    if len(ids)==0:return 0.0
    previous=depot
    first=-1
    cost=0
    load=0
    clock=initial_clock
    for index in range(len(ids)):
        unit=ids[index]
        reverse=int(reverses[index])
        entry=endpoints[unit,reverse]
        exit_node=endpoints[unit,1-reverse]
        if first<0:first=entry
        load+=demands[unit]
        if capacity and load>capacity:return np.inf
        if temporal:
            duration,ready,latest,valid=profiles[unit,reverse]
            arrival=clock+edge_distance(points,previous,entry)
            if not valid or arrival>latest:return np.inf
            clock=max(arrival+duration,ready)
            if clock+edge_distance(points,exit_node,depot)>depot_end:return np.inf
        cost+=internal[unit]
        if previous>=0:cost+=edge_distance(points,previous,entry)
        previous=exit_node
    cost+=edge_distance(points,previous,first if depot<0 else depot)
    return float(cost)
