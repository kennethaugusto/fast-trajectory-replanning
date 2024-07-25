import numpy as np
import random
import heapq

# Left, Up, Right, and Down, Constants used for Repeated Backwards A* # 
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Node class used for Repeated Forward and Adaptive A* #
class Node:
    def __init__(self, coords, parent=None, g=0, h=0):
        self.coords = coords
        self.parent = None
        self.g = g
        self.h = h

    def __lt__(self, other):
        if self.g == other.g:
            if random.random() > .5:
                return self.g > other.g
            else:
                return self.g < other.g
        elif self.g > other.g:
            #MAX
            return self.g > other.g
        else:
            return self.g < other.g
        
# Helper method used for Repeated Backwards A* that determines if a coordinate is within the bounds of the maze #        
def is_valid(x, y, grid_shape):
    rows, cols = grid_shape
    return 0 <= x < rows and 0 <= y < cols

# Helper method used for Adaptive A* that updates values needed for searching
def update_heuristics(closed_list, goal_g):
    for node in closed_list:
        node.h = goal_g - node.g

# Helper method used for Repeated Backwards A* that determines the manhattan distance of the current cell to the end # 
def man_dist(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)
        
# dfs algorithm used to generate mazes from numpy arrays #   
def dfs(grid, random_row, random_col, stack):
    neighbors = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, 1), (1, -1), (-1, -1)]
    while len(stack) != 0:
        random_neighbor = random.randint(0, len(neighbors) - 1)
        visited = set()
        while neighbors[random_neighbor][0] + random_row >= len(grid) or neighbors[random_neighbor][
            1] + random_col >= len(grid) or neighbors[random_neighbor][0] + random_row < 0 or \
                neighbors[random_neighbor][1] + random_col < 0 or grid[neighbors[random_neighbor][0] + random_row][
            neighbors[random_neighbor][1] + random_col] != 0:
            visited.add(random_neighbor)
            random_neighbor = random.randint(0, len(neighbors) - 1)
            if len(visited) == len(neighbors):
                stack.pop()
                if len(stack) != 0:
                    random_row = stack[len(stack) - 1][0]
                    random_col = stack[len(stack) - 1][1]
                break
        if len(visited) != len(neighbors):
            random_row = random_row + neighbors[random_neighbor][0]
            random_col = random_col + neighbors[random_neighbor][1]
            random_num = random.random()
            if random_num > .3:
                grid[random_row][random_col] = 1
            else:
                grid[random_row][random_col] = -1
            stack.append([random_row, random_col])

# Initializes 50 numpy arrays and random start point for dfs function # 
def create_grids():
    all_grids = np.zeros((50, 101, 101))
    for x in range(len(all_grids)):
        print('Generating grid',(x + 1))
        random_row = random.randint(0, all_grids.shape[1] - 1)
        random_col = random.randint(0, all_grids.shape[1] - 1)
        stack = []
        all_grids[x][random_row][random_col] = 1
        stack.append([random_row, random_col])
        dfs(all_grids[x], random_row, random_col, stack)
    return all_grids

# Repeated Forward A* algorithm running MAX by default #
def repeated_forward_a_star(grid, start_cord, end_cord):
    neighbors = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    open_list = []
    g_val = np.ones(grid.shape) * np.inf
    g_val[start_cord[0]][start_cord[1]] = 0
    g_n = 0
    h_n = np.sum(np.abs(end_cord - start_cord))
    f_n = g_n + h_n
    node = Node([start_cord[0], start_cord[1]], None, g_n, h_n)
    heapq.heappush(open_list, (f_n, node))
    closed_list = []
    while len(open_list) != 0:
        f_value, list_node = heapq.heappop(open_list)
        parent_x = list_node.coords[0]
        parent_y = list_node.coords[1]
        parent_g = list_node.g
        closed_list.append([parent_x, parent_y])
        if parent_x == end_cord[0] and parent_y == end_cord[1]:
            path = []
            while list_node:
                path.append(list_node.coords)
                list_node = list_node.parent
            path.reverse()
            return path

        for x, y in neighbors:
            new_x = parent_x + x
            new_y = parent_y + y
            if new_x == -1 or new_y == -1 or new_x == len(grid) or new_y == len(grid):
                pass
            else:
                if grid[new_x][new_y] == -1:
                    pass
                else:
                    g_n = parent_g + 1
                    if np.inf == g_val[new_x, new_y]:
                        g_val[new_x, new_y] = g_n
                        h_n = np.sum(np.abs(end_cord - np.array([new_x, new_y])))
                        f_n = g_n + h_n
                        new_node = Node([new_x, new_y], list_node, g_n, h_n)
                        new_node.parent = list_node
                        heapq.heappush(open_list, (f_n, new_node))

    return []

# Repeated Backwards A* algorithm #
def repeatedBackwardsA(grid, start, goal):
    
    num_rows, num_cols = grid.shape
    pq = []
    heapq.heappush(pq, (0, goal))
    g_val = np.ones(grid.shape) * np.inf
    g_val[goal] = 0
    f_val = np.ones(grid.shape) * np.inf 
    f_val[goal] = 0  
    parents = {}

    while pq:
        current_f, (current_x, current_y) = heapq.heappop(pq)

        if (current_x, current_y) == start:
            path = []
            while (current_x, current_y) in parents:
                path.append((current_x, current_y))
                current_x, current_y = parents[(current_x, current_y)]
            path.append(start)
            return path[::-1]

        for dx, dy in MOVES:
            neighbor_x, neighbor_y = current_x + dx, current_y + dy
            if is_valid(neighbor_x, neighbor_y, (num_rows, num_cols)) and grid[neighbor_x, neighbor_y] != -1:
                new_g = g_val[current_x, current_y] + 1
                new_h = man_dist(neighbor_x, neighbor_y, start[0], start[1])
                new_f = new_g + new_h
                if new_f < f_val[neighbor_x, neighbor_y]:
                    g_val[neighbor_x, neighbor_y] = new_g
                    f_val[neighbor_x, neighbor_y] = new_f
                    heapq.heappush(pq, (new_f, (neighbor_x, neighbor_y)))
                    parents[(neighbor_x, neighbor_y)] = (current_x, current_y)
                elif new_f == f_val[neighbor_x, neighbor_y] and new_g < g_val[neighbor_x, neighbor_y]:
                    g_val[neighbor_x, neighbor_y] = new_g
                    f_val[neighbor_x, neighbor_y] = new_f
                    heapq.heappush(pq, (new_f, (neighbor_x, neighbor_y)))
                    parents[(neighbor_x, neighbor_y)] = (current_x, current_y)
                elif new_f == f_val[neighbor_x, neighbor_y] and new_g == g_val[neighbor_x, neighbor_y]:
                    if random.random() < 0.5:
                        g_val[neighbor_x, neighbor_y] = new_g
                        f_val[neighbor_x, neighbor_y] = new_f
                        heapq.heappush(pq, (new_f, (neighbor_x, neighbor_y)))
                        parents[(neighbor_x, neighbor_y)] = (current_x, current_y)

    return None

# Adaptive A* Algorithm *
def adaptive_a_star(grid, start_cord, end_cord):
    neighbors = [(1, 0), (-1, 0), (0, -1), (0, 1)]
    open_list = []
    g_val = np.ones(grid.shape) * np.inf
    g_val[start_cord[0]][start_cord[1]] = 0
    g_n = 0
    closed_list = []
    closed_set = set()
    f_val = np.ones(grid.shape) * np.inf

    h_n = np.sum(np.abs(np.array(end_cord) - np.array(start_cord)))
    f_n = g_n + h_n

    start_node = Node([start_cord[0], start_cord[1]], None, g_n, h_n)
    heapq.heappush(open_list, (f_n, start_node))

    while len(open_list) != 0:
        _, current_node = heapq.heappop(open_list)
        current_coords = tuple(current_node.coords)

        if current_coords == tuple(end_cord):
            update_heuristics(closed_list, current_node.g)
            path = []
            while current_node:
                path.append(current_node.coords)
                current_node = current_node.parent
            return path[::-1]
        
        closed_list.append(current_node)
        closed_set.add(current_coords)

        for dx, dy in neighbors:
            new_x, new_y = current_node.coords[0] + dx, current_node.coords[1] + dy
            new_coords = (new_x, new_y)

            if is_valid(new_x, new_y, grid.shape) and grid[new_x][new_y] != -1 and new_coords not in closed_set:
                new_g = g_val[current_node.coords[0], current_node.coords[1]] + 1
                new_h = man_dist(new_x, new_y, current_node.coords[0], current_node.coords[1])
                new_f = new_g + new_h
                g_n = current_node.g + 1
                if new_f < f_val[new_x, new_y]:
                    g_val[new_x, new_y] = g_n
                    h_n = np.sum(np.abs(np.array(end_cord) - np.array([new_x, new_y])))
                    f_n = g_n + h_n
                    new_node = Node([new_x, new_y], current_node, g_n, h_n)
                    new_node.parent = current_node
                    heapq.heappush(open_list, (f_n, new_node))
                    closed_set.add(new_coords)

    return []

# Driver for Repeated Forward A* algorithm, ensures correct grid start and end points if blocked and prints solved grids #
def runRFA(grids, start, goal):
    
    unsolved = 0
    rows, cols = grids[0].shape
    for g in range(len(grids)):

        if grids[g][start[0]][start[1]] == -1:
            grids[g][start[0]][start[1]] = 1
        if grids[g][goal[0]][goal[1]] == -1:
            grids[g][goal[0]][goal[1]] = 1
        
        path = repeated_forward_a_star(grids[g], start, goal)
        if len(path) == 0:
            unsolved += 1
            continue

        for x in range(rows):
            for y in range(cols):
                if grids[g][x][y] == -1:
                    print("x", end=' ')
                elif grids[g][x][y] == 1:
                    if [x, y] in path:
                        print(".", end=' ')
                    else:
                        print("o", end=' ')
            print()
        
        print()
        print()
    
    print("Total Grids Solved:", len(grids) - unsolved)
    print("Total Grids Unsolved:", unsolved)

# Driver for Repeated Backwards A* algorithm, ensures correct grid start and end points if blocked and prints solved grids #
def runRBA(grids, start, goal):
    
    unsolved = 0
    rows, cols = grids[0].shape
    for g in range(len(grids)):

        if grids[g][start[0]][start[1]] == -1:
            grids[g][start[0]][start[1]] = 1
        if grids[g][goal[0]][goal[1]] == -1:
            grids[g][goal[0]][goal[1]] = 1

        path = repeatedBackwardsA(grids[g], start, goal)
        if path is None:
            unsolved += 1
            continue

        for x in range(rows):
            for y in range(cols):
                if grids[g][x][y] == -1:
                    print("x", end=' ')
                elif grids[g][x][y] == 1:
                    if (x, y) in path:
                        print(".", end=' ')
                    else:
                        print("o", end=' ')
            print()

        print()
        print()

    print("Total Grids Solved:", len(grids) - unsolved)
    print("Total Grids Unsolved:", unsolved)

# Driver for Adaptive A* algorithm, ensures correct grid start and end points if blocked and prints solved grids #
def runADA(grids, start, goal):

    unsolved = 0
    rows, cols = grids[0].shape
    for g in range(len(grids)):

        if grids[g][start[0]][start[1]] == -1:
            grids[g][start[0]][start[1]] = 1
        if grids[g][goal[0]][goal[1]] == -1:
            grids[g][goal[0]][goal[1]] = 1

        path = adaptive_a_star(grids[g], start, goal)
        if path is None:
            unsolved += 1
            continue

        if len(path) == 0:
            unsolved += 1
            continue

        for x in range(rows):
            for y in range(cols):
                if grids[g][x][y] == -1:
                    print("x", end=' ')
                elif grids[g][x][y] == 1:
                    if [x, y] in path:
                        print(".", end=' ')
                    else:
                        print("o", end=' ')
            print()

        print()
        print()
    
    print("Total Grids Solved:", len(grids) - unsolved)
    print("Total Grids Unsolved:", unsolved)

# Demo code that runs until a correct path is found and prints emojis for a grid for visual clarity #
def demoCodeE():

    path = None
    while path is None:

        demoG = np.zeros((10,10))

        for x in range(0, 10):
            for y in range(0, 10):
                if random.random() <= 0.3:
                    demoG[x][y] = -1
                else:
                    demoG[x][y] = 1

        start = (2, 3)
        end = (9, 9)

        
        path = repeatedBackwardsA(demoG, start, end)
        if path is None: continue

        print("\nEmoji Grid\n")

        for i in range(0, 10):
            for j in range(0, 10):
                if demoG[i][j] == -1:
                    print("⬛", end=' ')
                elif demoG[i][j] == 1:
                    if (i, j) in path:
                        print("🟨", end=' ')
                    else:
                        print("⬜", end=' ')
            print()

        print()

# Demo Code that runs until correct path is found and prints single 101x101 grid using Repeated Backwards A* # 
def demoCode():

    path2 = None
    while(path2 is None):
        print("Running until valid grid is found")

        temp = create_grids()
        demoGrid2 = temp[0]

        start = (0, 0)
        end2 = (100, 100)

        
        path2 = repeatedBackwardsA(demoGrid2, start, end2)
        if path2 is None: continue

        print("Bigger grid:\n")

        for k in range(0, 101):
            for l in range(0, 101):
                if demoGrid2[k][l] == -1:
                    print("x", end=' ')
                elif demoGrid2[k][l] == 1:
                    if (k, l) in path2:
                        print(".", end=' ')
                    else:
                        print("o", end=' ')
            print()

        print()


# Driver main function for running code #
if __name__ == "__main__":

    grids = create_grids()
    #print paths with .'s instead of emojis
    #create switch for options to run forward, backward, and adaptive, and demo
    print("\nSelect which algorithm to run:")
    choice = int(input(" 1 - Repeated Forward A*\n 2 - Repeated Backwards A*\n 3 - Adaptive A*\n 4 - Demo Code\n"))

    match choice:
        case 1:
            #start coordinates
            start_cord = np.array((0, 0))
            #end coordinates
            end_cord = np.array((100, 100))
            #Run Repeated Forward A*
            runRFA(grids,start_cord, end_cord)
        case 2:
            #start/end coordinates are a set instead of an array for the other algorithms
            start_cord = (0, 0)
            #end coordinates
            end_cord = (100, 100)
            #Run Repeated Backwards A*
            runRBA(grids, start_cord, end_cord)
        case 3:
            #start coordinates
            start_cord = (0, 0)
            #end coordinates
            end_cord = (100, 100)
            #Run Adaptive A*
            runADA(grids, start_cord, end_cord)
        case 4:
            ec = input("emojis?(y/n)\n")
            if ec == "y":
                demoCodeE()
            else:
                demoCode()
        case defualt:
            print("Not a valid number")
    