from A_star_search import *
from RRT import *
from PIL import Image

# The start and goal coodinates
start = (635, 140)
goal = (350, 400)

# Convert image into a numpy array. 1 indicates free space, 0 indicates obstacle
occupancy_map_img = Image.open('occupancy_map.png')
occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)

# Let's test the reachability
'''print("Check if it is possible to get from start point to goal (should be false):")
print(is_reachable(occupancy_grid, start, goal))
print("\nCheck if it is possible to get from a start point to a neighbor (should be true):")
print(is_reachable(occupancy_grid, start, (640,145)))'''

# Construct an A* path
path_a = find_a_path(occupancy_grid, start, goal)
x_path_a, y_path_a = zip(*path_a)

# Construct an RRT* path
path_rrt = rrt_search(occupancy_grid, start, goal, 20000, 10)
x_path_rrt, y_path_rrt = zip(*path_rrt)

plt.scatter(y_path_a, x_path_a, s=1, color='orange', label='A* search')
plt.plot(y_path_rrt, x_path_rrt, color='green', label='RRT search')
plt.scatter(start[1], start[0], s=50, color='r')
plt.scatter(goal[1], goal[0], s=50, color='b')
plt.imshow(occupancy_grid, cmap='gray')
plt.legend()
plt.savefig('path.png')
plt.show()


