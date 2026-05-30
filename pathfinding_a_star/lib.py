import heapq

def manhatten_distance(start, end):
    return abs(start[0]-end[0]) + abs(start[1]-end[1])

def find_neighbour(position):
    neighbours = []
    path_to_move = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    for move in path_to_move:
        neighbour_pos = (position[0] + move[0] , position[1] + move[1])

def valid_neighbour(maze, neighbour_pos, close_list):
    row, col = neighbour_pos
    value = maze[row][col]
    if value == 1:
        # This is an obstacle
        return False
    elif neighbour_pos in close_list:
        return False
    elif not (0 <= row < len(maze)) and (0 <= col < len(maze[0])):
        return False
    else:
        return True

def a_star_pathfinding(start, end, maze):
    """ Executes the a star pathfinding to find the best path from start to end
    """
    open = []
    close = []

    # Add start node to open
    heapq.heappush(open, start)

    # Calculate the scores
    g_score = {start: 0}
    f_score = {start: manhatten_distance(start, end)}








if __name__ == "__main__":
    maze = [
        [0,0,1,0,0,1],
        [1,0,1,0,0,1],
        [0,0,0,0,1,0],
        [0,1,1,0,1,0],
        [0,0,0,0,0,0],
    ]
    start = (0,0)
    end = (5,2)
    close_list = []
    print(len(maze[0]))
    print(valid_neighbour(maze, (0,1), close_list))