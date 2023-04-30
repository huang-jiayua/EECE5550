#!/usr/bin/env python
# coding: utf-8
# In[1]:
# Load the libraries
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from bresenham import bresenham

# Read image from disk using PIL
occupancy_map_img = Image.open(r'C:\Users\froze\Desktop\EECE 5550\HW2\occupancy_map.png')
# Interpret this image as a numpy array, and threshold its values to,→ {0,1}
M = (np.asarray(occupancy_map_img) > 0).astype(int)


# Function to get the Euclidean distance between 2 vertices
def d(v1, v2):
    r1, c1 = v1[0], v1[1]
    r2, c2 = v2[0], v2[1]
    dis = math.dist([r1, c1], [r2, c2])
    return dis


# Function to generate a vertex uniformly and randomly from the free space of M
def sample_vertex(M0):
    global V
    while True:
        # Samples are uniformly distributed over the half-open interval [low, high)
        v_new = (int(np.random.uniform(low=0, high=M0.shape[0])),
                 int(np.random.uniform(low=0, high=M0.shape[1])))
        # When vertex is not selected before
        if v_new not in V:
            # When vertex is not occupied
            if M0.item(v_new) == 1:
                return v_new


# Function to check if the path from v1 to v2 is obstacle-free
# returns TRUE when not obstacle in the straight line path
def path_free(M1, v1, v2):
    # Creates a straight-line path between 2 points
    path = list(bresenham(v1[0], v1[1], v2[0], v2[1]))
    for v in path:
        # when cell in the path is occupied
        if M1.item(v) == 0:
            return False
        return True


# Function to add a new vertex vnew to the PRM
def add_vertex(M2, vnew, dmax1):
    global V

    # adds a node to new position in node list and the graph
    V.append(vnew)
    PRMG.add_node(PRMG.number_of_nodes() + 1, pos=vnew)

    for i1 in range(len(V)):
        v = V[i1]

        # When the vnew is not selected before and the distance between
        # the v and vnew <= local search radius
        if vnew != v and d(v, vnew) <= dmax1:
            # When the path between v and vnew is obstacle free
            if path_free(M2, v, vnew):
                PRMG.add_edge(i1 + 1, PRMG.number_of_nodes(), weight=d(v, vnew))
    return PRMG


# Function to contract the PRM using the occupancy grid,
# number of samples and local search radius
def construct_prm(M3, N3, dmax2):
    global V
    # for the counter less than Maximum allowed samples
    for k in range(N3):
        # sample a point uniformly and randomly from M
        vnew = sample_vertex(M3)
        # add the point to PRM
        add_vertex(M3, vnew, dmax2)
    return PRMG


# Initialize variables
V = list()
PRMG = nx.Graph()
N = 2500  # maximum number of samples
dmax = 75  # maximum local search radius
# Construct the PRM using the occupancy grid,
# maximum sample allowed and max local search radius
PRMG = construct_prm(M, N, dmax)

inv_pos = {}
# dictionary of all the node position in the graph
pos = nx.get_node_attributes(PRMG, 'pos')
# changing the origin to be in top left corner
for i in range(1, PRMG.number_of_nodes() + 1):
    x = -pos.get(i)[0]
    y = pos.get(i)[1]
    inv_pos[i] = (y, x)
# Plotting the PRM
fig = plt.figure(1, figsize=(150, 150), dpi=60)

nx.draw_networkx(PRMG, inv_pos, node_size=50, linewidths=0.3, with_labels=False)
