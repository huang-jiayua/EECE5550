import heapq
import networkx as nx
import numpy
import math
import matplotlib.pyplot
from bresenham import bresenham
import random

from PIL import Image

occupancy_map_img = Image.open(r'C:\Users\froze\Desktop\EECE 5550\HW2\occupancy_map.png')

occupancy_grid = (numpy.asarray(occupancy_map_img) > 0).astype(int)


def sample(G):
    row, col = G.shape
    sample1 = random.uniform(0, row)
    sample2 = random.uniform(0, col)
    while G.item(int(sample1), int(sample2)) == 0:
        sample1 = random.uniform(0, row)
        sample2 = random.uniform(0, col)
    out = (int(sample1), int(sample2))
    return out


def valid_Path(G, local_start, local_goal):
    for v in list(bresenham(local_start[0], local_start[1], local_goal[0], local_goal[1])):
        if G.item(v) == 0:
            return False
    return True


def d(v1, v2):
    return math.dist([v1[0], v1[1]], [v2[0], v2[1]])


def add_Vertex(G, vnew, dmax_vert):
    global V, E
    V.append(vnew)
    # Nodes named after its number, with a pos attr that record its place in the graph
    # print(PRM.number_of_nodes() + 1, vnew)
    PRM.add_node(PRM.number_of_nodes() + 1, pos=vnew)

    for index in range(len(V)):
        current = V[index]
        if current != vnew and d(current, vnew) < dmax_vert and valid_Path(G, current, vnew):
            PRM.add_edge(index + 1, PRM.number_of_nodes(), weight=d(current, vnew))
            E.append((current, vnew))
    return PRM


def construct_PRM(G, Net, dmax_cons):
    global V, E
    # V = None
    # E = None
    V = list()
    E = list()
    for _ in range(Net):
        # if vnew not in G
        #   break
        # Implemented inside the sample function
        vnew = sample(G)
        add_Vertex(G, vnew, dmax_cons)
    return PRM


V = list()
E = list()
N = 2500
dmax = 75
start = (635, 140)
goal = (350, 400)

PRM = nx.Graph()
PRM_complete = construct_PRM(occupancy_grid, N, dmax)

# Extract pos attr prom the PRM constructed
pos_list = nx.get_node_attributes(PRM_complete, 'pos')

# print(pos_list.get(1))

repos_list = {}

# Inverse the graph's coordinate to fit the original graph
for i in range(1, PRM_complete.number_of_nodes() + 1):
    inv_X = -pos_list.get(i)[0]
    inv_Y = pos_list.get(i)[1]
    repos_list[i] = (inv_Y, inv_X)

output_graph = matplotlib.pyplot.figure(1, figsize=(150, 150), dpi=60)
nx.draw_networkx(PRM_complete, repos_list, node_size=100, linewidths=0.1, with_labels=False)

add_Vertex(occupancy_grid, start, dmax)
add_Vertex(occupancy_grid, goal, dmax)

startIndex = PRM_complete.number_of_nodes() - 1
goalIndex = PRM_complete.number_of_nodes()

# Find the index of start and goal (in case they are already added)
for t in range(PRM_complete.number_of_nodes()):
    if V[t] == start:
        startIndex = t + 1
    if V[t] == goal:
        goalIndex = t + 1

# Use A* Algorithm provided on the two new nodes
A_star = nx.astar_path(PRM_complete, startIndex, goalIndex)
length = nx.astar_path_length(PRM_complete, startIndex, goalIndex)

print("Sum of length taken is: ", length)

path = list()
for e in A_star:
    path.append(PRM_complete.nodes[e]['pos'])

# print(path)

matplotlib.pyplot.imshow(matplotlib.pyplot.imread(
    r'C:\Users\froze\Desktop\EECE 5550\HW2\occupancy_map.png'), extent=[0, 620, 680, 0])

matplotlib.pyplot.plot(numpy.array(path)[:, 1], numpy.array(path)[:, 0], 'b')

matplotlib.pyplot.show()
