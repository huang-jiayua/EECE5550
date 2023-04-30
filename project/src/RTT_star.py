import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from bresenham import bresenham
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import distance
import random


# from sklearn.preprocessing import normalize


class Node:
    def __init__(self, pos, parent):
        self.pos = pos
        self.cost = np.inf
        self.parent = None


def rrt_star_search(map, start, end, n, step):
    def is_collision_free(p1, p2, occupancy_map):
        # Check if the starting and ending points are the same
        if (p1 == p2).all():
            return False

        # for v in list(bresenham(p1[0], p1[1], p2[0], p2[1])):
        #     if occupancy_map.item(v) == 0:
        #         return False
        # return True

        # Get the x and y coordinates of the two points
        x1, y1 = p1
        x2, y2 = p2

        # Check if the starting or ending points are inside an obstacle
        if occupancy_map[x1, y1] == 0 or occupancy_map[x2, y2] == 0:
            return False

        # Determine the step sizes for x and y
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        # Determine the initial error
        error = dx - dy

        # Loop through the occupancy map until the two points are reached
        while x1 != x2 or y1 != y2:
            # Check if there is an obstacle at the current point
            if occupancy_map[x1, y1] == 0:
                return False

            # Determine the next point to check
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x1 += sx
            if e2 < dx:
                error += dx
                y1 += sy

        # There are no obstacles between the two points and neither of the points are inside an obstacle
        return True

    def get_random_point(map):
        free_points = np.argwhere(map == 1)
        return tuple(random.choice(free_points))

    def get_nearest_node(point, graph):
        nearest_node, min_dist = None, float('inf')
        for node in graph.nodes:
            dist = distance.euclidean(point, node.pos)
            if dist < min_dist:
                nearest_node, min_dist = node, dist
        return nearest_node

    def get_new_point(nearest_node, random_point, step):
        direction = np.array(random_point) - np.array(nearest_node)
        if np.linalg.norm(direction) == 0:
            print("nearest_node: ", nearest_node, "random_point: ", random_point)
        direction = direction / np.linalg.norm(direction)
        new_point = np.array(nearest_node) + step * direction

        # Ensure that new_point is within the valid bounds of the map
        new_point[0] = max(0, min(map.shape[0] - 1, new_point[0]))
        new_point[1] = max(0, min(map.shape[1] - 1, new_point[1]))

        # convert new_point into a Node
        new_node = Node(new_point.astype(int), None)

        return new_node

    # Initialize V and E
    G = nx.Graph()
    start_node = Node(start, None)
    start_node.cost = 0
    G.add_node(start_node)

    end_node_added = False
    radius = step + 1
    # Start the loop
    for i in range(n):
        progress = (i / n) * 100

        # Print progress as percentage with 1 decimal place and only one line
        print("\rProgress: {:.1f}%".format(progress), end="")

        # Get a random sample from map (occupancy map)
        random_point = get_random_point(map)

        # Get nearest neighbor of q_target
        nearest_node = get_nearest_node(random_point, G)

        if (nearest_node.pos[0] == random_point[0]) and (nearest_node.pos[1] == random_point[1]):
            continue

        # Create a new point
        new_point = get_new_point(nearest_node.pos, random_point, step)

        if is_collision_free(nearest_node.pos, new_point.pos, map):
            new_point.parent = nearest_node
            new_point.cost = nearest_node.cost + distance.euclidean(nearest_node.pos, new_point.pos)
            # Find all the nodes within a radius of the new point
            neighbors = [Node for Node in G.nodes if distance.euclidean(Node.pos, new_point.pos) <= radius]

            # Find the neighbor with the lowest cost
            min_cost = new_point.cost
            min_cost_node = nearest_node
            for neighbor in neighbors:
                cost = neighbor.cost + distance.euclidean(neighbor.pos, new_point.pos)
                if cost < min_cost:
                    min_cost = cost
                    min_cost_node = neighbor

            # If there is a neighbor with a lower cost, set the new point's parent to that neighbor
            if min_cost_node is not nearest_node:
                new_point.parent = min_cost_node
                new_point.cost = min_cost

            # Update the cost of all the neighbors
            for neighbor in neighbors:
                cost = new_point.cost + distance.euclidean(new_point.pos, neighbor.pos)
                if cost < neighbor.cost:
                    neighbor.cost = cost
                    neighbor.parent = new_point
                    G.add_edge(new_point, neighbor)

            # Add the new point to the graph
            G.add_edge(nearest_node, new_point)
            G.add_node(new_point)

            # Check if the new point is close enough to the goal and update end's parent if it is
            if distance.euclidean(new_point.pos, end) <= step and is_collision_free(new_point.pos, end, map):
                if not end_node_added:
                    end_node = Node(end, new_point)
                    end_node.cost = new_point.cost + distance.euclidean(new_point.pos, end)
                    G.add_node(end_node)
                    G.add_edge(new_point, end_node)
                    end_node_added = True
                if new_point.cost + distance.euclidean(new_point.pos, end) < end_node.cost:
                    end_node.parent = new_point
                    end_node.cost = new_point.cost + distance.euclidean(new_point.pos, end)
                    G.add_edge(new_point, end_node)

            # if is_collision_free(new_point, end, map):
            #     G.add_edge(new_point, end)
            #     G.add_node(end)
            #     break

    # Get the path from the start to the end using parent pointers
    path = []
    current_node = end_node
    while current_node is not None:
        path.append(current_node.pos)
        current_node = current_node.parent
    path.reverse()
    return path

    # if nx.has_path(G, start, end):
    #     return nx.shortest_path(G, start, end)
    # else:
    #     print('No path was found')
    #     return G


# ============= Implementation of the code =============
# The start and goal coodinates
start = (635, 140)
goal = (350, 400)

# Convert image into a numpy array. 1 indicates free space, 0 indicates obstacle
occupancy_map_img = Image.open('occupancy_map.png')
occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)

# Construct an RRT* path
path_rrt_star = rrt_star_search(occupancy_grid, start, goal, 10000, 10)
print(path_rrt_star)

x, y = zip(*path_rrt_star)

# plt.scatter(y, x, s=1, color='orange', label='RRT search')
plt.plot(y, x, color='orange', label='RRT* search')
plt.scatter(start[1], start[0], s=50, color='r')
plt.scatter(goal[1], goal[0], s=50, color='b')
plt.imshow(occupancy_grid, cmap='gray')
plt.legend()
plt.show()
