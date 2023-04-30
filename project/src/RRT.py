import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial import distance
import random
from sklearn.preprocessing import normalize


def rrt_search(map, start, end, n, step):
    def is_collision_free(p1, p2, occupancy_map):
        # Check if the starting and ending points are the same
        if p1 == p2:
            return False

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
            dist = distance.euclidean(point, node)
            if dist < min_dist:
                nearest_node, min_dist = node, dist
        return nearest_node

    def get_new_point(nearest_node, random_point, step):
        direction = np.array(random_point) - np.array(nearest_node)
        direction = direction / np.linalg.norm(direction)
        new_point = np.array(nearest_node) + step * direction

        # Ensure that new_point is within the valid bounds of the map
        new_point[0] = max(0, min(map.shape[0] - 1, new_point[0]))
        new_point[1] = max(0, min(map.shape[1] - 1, new_point[1]))

        return tuple(new_point.astype(int))

    # Initialize V and E
    G = nx.Graph()
    G.add_node(start)

    # Start the loop
    for _ in range(n):
        # Get a random sample from map (occupancy map)
        random_point = get_random_point(map)

        # Get nearest neighbor of q_target
        nearest_node = get_nearest_node(random_point, G)

        # Create a new point
        new_point = get_new_point(nearest_node, random_point, step)

        if is_collision_free(nearest_node, new_point, map):
            G.add_edge(nearest_node, new_point)
            G.add_node(new_point)
            if is_collision_free(new_point, end, map):
                G.add_edge(new_point, end)
                G.add_node(end)
                break

    if nx.has_path(G, start, end):
        return nx.shortest_path(G, start, end)
    else:
        print('No path was found')
        return G


'''# ============= Implementation of the code =============
# The start and goal coodinates
start = (635, 140)
goal = (350, 400)

# Convert image into a numpy array. 1 indicates free space, 0 indicates obstacle
occupancy_map_img = Image.open('occupancy_map.png')
occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)


# Construct an RRT* path
path_rrt = rrt_search(occupancy_grid, start, goal, 20000, 10)
print(path_rrt)


x, y = zip(*path_rrt)

#plt.scatter(y, x, s=1, color='orange', label='RRT search')
plt.plot(y, x, color='orange', label='RRT search')
plt.scatter(start[1], start[0], s=50, color='r')
plt.scatter(goal[1], goal[0], s=50, color='b')
plt.imshow(occupancy_grid, cmap='gray')
plt.legend()
plt.show()'''



