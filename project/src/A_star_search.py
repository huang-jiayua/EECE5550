#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import heapq as hq
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import math
from PIL import Image
from queue import PriorityQueue
from heapq import heappush, heappop

def is_reachable(M, v1, v2):
    row1, col1 = v1
    row2, col2 = v2

    # Check if either one of the vertices are occupied
    if M[row1][col1] == 0 or M[row2][col2] == 0:
        return False

    # Check if line segment intersects with any occupied cells
    delta_row = row2 - row1
    delta_col = col2 - col1

    if abs(delta_row) >= abs(delta_col):
        steps = abs(delta_row)
        for i in range(steps + 1):
            r = row1 + int(i * delta_row / steps)
            c = col1 + int(i * delta_col / steps)
            if M[r][c] == 0:
                return False
    else:
        steps = abs(delta_col)
        for i in range(steps + 1):
            r = row1 + int(i * delta_row / steps)
            c = col1 + int(i * delta_col / steps)
            if M[r][c] == 0:
                return False
    return True

def recover_path(s, g, pred):
    path = [g]
    while path[-1] != s:
        path.append(pred[path[-1]])
    path.reverse()
    return path

def neighbor_nodes(v, occupancy_grid):
    x, y = v
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1), (x+1, y+1), (x-1, y-1), (x-1, y+1), (x+1, y-1)]
    neighbors = [n for n in neighbors if 0 <= n[0] < occupancy_grid.shape[0] and 0 <= n[1] < occupancy_grid.shape[1]]
    return [n for n in neighbors if occupancy_grid[n[0], n[1]] == 1]

def heuristic_cost(u, v, occupancy_grid):
    x1, y1 = u
    x2, y2 = v
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def edge_weight(u, v, occupancy_grid):
    return heuristic_cost(u, v, occupancy_grid)

def a_star_search(V, s, g, N, w, h):
    CostTo = {}
    EstTotalCost = {}
    pred = {}
    for v in V:
        CostTo[v] = float('inf')
        EstTotalCost[v] = float('inf')
    CostTo[s] = 0
    EstTotalCost[s] = h(s, g)
    Q = [(EstTotalCost[s], s)]

    # Main loop,. while we have nods in queue
    while Q:
        # Pop the vertex with the lowest estimated total cost from Q
        _, v = heappop(Q)

        # If the popped vertex is the goal, return the optimal path
        if v == g:
            return recover_path(s, g, pred)

        # For each of v's neighbors
        for i in N(v):
            # Calculate the cost of the path to reach i through v
            pvi = CostTo[v] + w(v, i)

            # If the path to reach i through v is better than the previously-known best path
            if pvi < CostTo[i]:
                # Update the parent of i to v
                pred[i] = v

                # Update the cost of the best path to i
                CostTo[i] = pvi

                # Update the estimated total cost of the best path to i
                EstTotalCost[i] = pvi + h(i, g)

                # If i is already in Q, update its priority
                if (EstTotalCost[i], i) in Q:
                    # Update i's priority in the queue
                    Q.remove((EstTotalCost[i], i))
                    heappush(Q, (EstTotalCost[i], i))
                else:
                    # Insert i into the queue with its EstTotalCost as its priority
                    heappush(Q, (EstTotalCost[i], i))

    # Return empty if there is no path to the goal
    return []

# main function that invokes A* search algorithm 
def find_a_path(occupancy_grid, start, goal):
    V = set((x, y) for x in range(len(occupancy_grid)) for y in range(len(occupancy_grid[0])))
    N = lambda v: neighbor_nodes(v, occupancy_grid)
    w = lambda u, v: edge_weight(u, v, occupancy_grid)
    h = lambda u, v: heuristic_cost(u, v, occupancy_grid)
    s = start
    g = goal
    path = a_star_search(V, s, g, N, w, h) 
    return path
