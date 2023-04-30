import heapq
import numpy
import math
import matplotlib.pyplot

from PIL import Image

occupancy_map_img = Image.open(r'C:\Users\froze\Desktop\EECE 5550\HW2\occupancy_map.png')

occupancy_grid = (numpy.asarray(occupancy_map_img) > 0).astype(int)


def recoverPath(start, goal, pred):
    length = list()
    trace_back = [goal]
    while goal != start:
        length.append(d(goal, pred[goal]))
        goal = pred[goal]
        trace_back.append(goal)
    trace_back = list(reversed(trace_back))
    return trace_back, sum(length)


def d(v1, v2):
    return math.dist([v1[0], v1[1]], [v2[0], v2[1]])


def N(v):
    row, col = v
    return (
        (row + g, col + h)
        for g in (-1, 0, 1)
        for h in (-1, 0, 1)
        if g != 0 or h != 0
        if 0 <= row + g & row + g < len(occupancy_grid)
        if 0 <= col + h & col + h < len(occupancy_grid[0])
        if occupancy_grid[row + g][col + h] == 1)


def A_star(Vertex, start, goal, Neighbor, weight, heuristic):
    # Initialization
    pred = dict()
    CostTo = dict()
    EstTotalCost = dict()
    Q = list()

    for v in Vertex:
        CostTo[v] = float('inf')
        EstTotalCost[v] = float('inf')

    CostTo[start] = 0
    EstTotalCost[start] = heuristic(start, goal)

    heapq.heappush(Q, (heuristic(start, goal), start))

    while Q:
        priority, v = heapq.heappop(Q)
        # If reached the goal, end the loop and start the trace back algorithm
        if v == goal:
            return recoverPath(start, goal, pred)
        for index in Neighbor(v):
            # print(f'{index}: {list(Neighbor(v))}')
            pvi = CostTo[v] + weight(v, index)
            if pvi < CostTo[index]:
                # Update based on heuristic
                pred[index] = v
                CostTo[index] = pvi
                EstTotalCost[index] = pvi + heuristic(index, goal)
                # insert here
                # heapq.has(index) ?
                # if any(index == b for a, b in Q):
                #     heapq.heapreplace()
                heapq.heappush(Q, (heuristic(index, goal), index))

    return None


V = list()

for i in range(0, len(occupancy_grid)):
    for j in range(0, len(occupancy_grid[1])):
        x = i, j
        V.append(x)

path, le = A_star(V, (635, 140), (350, 400), N, d, d)

print("Sum of length taken is: ", le)

# print(list(N((0, 0))))

matplotlib.pyplot.imshow(matplotlib.pyplot.imread(r'C:\Users\froze\Desktop\EECE 5550\HW2\occupancy_map.png'),
                         extent=[0, 620, 680, 0])
matplotlib.pyplot.plot(numpy.array(path)[:, 1], numpy.array(path)[:, 0], 'b')

matplotlib.pyplot.show()
