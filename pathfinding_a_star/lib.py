import heapq

class a_star_pathfinding():
    def __init__(self, maze, start, end):
        self.maze = maze
        self.start = start
        self.end = end
        self.open_list = []
        self.close_list = set()
        self.parent = {}

    def manhatten_distance(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def find_neighbour(self, position):
        neighbours = []
        path_to_move = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
        for move in path_to_move:
            neighbour_pos = (position[0] + move[0] , position[1] + move[1])
            if self.valid_neighbour(neighbour_pos):
                neighbours.append(neighbour_pos)
        return neighbours

    def valid_neighbour(self, neighbour_pos):
        row, col = neighbour_pos
        if not(0 <= row < len(self.maze)):
            return False
        elif not(0 <= col < len(self.maze[0])):
            return False
        elif self.maze[row][col] == 0:
            # This is an obstacle
            return False
        elif neighbour_pos in self.close_list:
            return False
        else:
            return True

    def clear(self):
        self.open_list.clear()
        self.close_list = set()

    def change_start_end(self, start, end):
        self.start = start
        self.end = end
    
    def change_maze(self, maze):
        self.maze = maze

    def pathfind(self):
        """ Executes the a star pathfinding to find the best path from start to end
        """
        # Empty list
        self.clear()

        # Add start node to open
        heapq.heappush(self.open_list, (0, self.start))

        # Calculate the scores
        g_score = 0
        f_score = self.manhatten_distance(start, end)
        total_score = g_score + f_score
        track_total_score = {self.start: (g_score, total_score)}

        while self.open_list:
            prio, current_pos = heapq.heappop(self.open_list)
            self.close_list.add(current_pos)

            # Target found
            if current_pos == self.end:
                return self.reconstruct_path(current_pos)

            neighbours = self.find_neighbour(current_pos)
            for neighbour_pos in neighbours:
                neighbour_g_score = track_total_score[current_pos][0] + 1
                neighbour_f_score = self.manhatten_distance(neighbour_pos, end)
                neighbour_total_score = neighbour_g_score + neighbour_f_score

                if neighbour_pos not in track_total_score or neighbour_total_score < track_total_score[neighbour_pos][1]:
                    self.parent[neighbour_pos] = current_pos
                    track_total_score[neighbour_pos] = (neighbour_g_score, neighbour_total_score)
                    if neighbour_pos not in self.open_list:
                        heapq.heappush(self.open_list, (track_total_score, neighbour_pos))
                
        return None
    
    def reconstruct_path(self, current_pos):
        best_path = [current_pos]
        while current_pos in self.parent:
            current_pos = self.parent[current_pos]
            best_path.append(current_pos)
        best_path.reverse()
        return best_path


if __name__ == "__main__":
    maze = [
        [1,1,0,1,1,0],
        [0,1,1,1,1,1],
        [0,1,0,0,1,0],
        [0,1,1,1,1,0],
        [0,0,0,0,1,0],
    ]
    start = (0,0)
    end = (4,4)
    
    pathfinder = a_star_pathfinding(maze, start, end)
    path = pathfinder.pathfind()
    if path:
        print(path)
    else:
        print("No valid path found")