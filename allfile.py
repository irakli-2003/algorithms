#adjList.py
def add_edge(adj, i, j):
    adj[i].append(j)
    adj[j].append(i)  # Undirected

def display_adj_list(adj):
    for i in range(len(adj)):
        print(f"{i}: ", end="")
        for j in adj[i]:
            print(j, end=" ")
        print()

# Create a graph with 4 vertices and no edges
V = 4
adj = [[] for _ in range(V)]

# Now add edges one by one
add_edge(adj, 0, 1)
add_edge(adj, 0, 2)
add_edge(adj, 1, 2)
add_edge(adj, 2, 3)

print("Adjacency List Representation:")
display_adj_list(adj)
#############################################################
#adjMatrix.py
def add_edge(mat, i, j):
    # Add an edge between two vertices
    mat[i][j] = 1  # Graph is
    mat[j][i] = 1  # Undirected

def display_matrix(mat):
    # Display the adjacency matrix
    for row in mat:
        print(" ".join(map(str, row)))

# Main function to run the program
if __name__ == "__main__":
    V = 4  # Number of vertices
    mat = [[0] * V for _ in range(V)]

    # Add edges to the graph
    add_edge(mat, 0, 1)
    add_edge(mat, 0, 2)
    add_edge(mat, 1, 2)
    add_edge(mat, 2, 3)

    # Optionally, initialize matrix directly
    """
    mat = [
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0]
    ]
    """

    # Display adjacency matrix
    print("Adjacency Matrix:")
    display_matrix(mat)
#######################################################
#bfs.py
from collections import deque


# Function to find BFS of Graph from given source s
def bfs(adj):
    # get number of vertices
    V = len(adj)

    # create an array to store the traversal
    res = []
    s = 0
    # Create a queue for BFS
    q = deque()

    # Initially mark all the vertices as not visited
    visited = [False] * V

    # Mark source node as visited and enqueue it
    visited[s] = True
    q.append(s)

    # Iterate over the queue
    while q:

        # Dequeue a vertex from queue and store it
        curr = q.popleft()
        res.append(curr)

        # Get all adjacent vertices of the dequeued
        # vertex curr If an adjacent has not been
        # visited, mark it visited and enqueue it
        for x in adj[curr]:
            if not visited[x]:
                visited[x] = True
                q.append(x)

    return res


if __name__ == "__main__":

    # create the adjacency list
    # [ [2, 3, 1], [0], [0, 4], [0], [2] ]
    adj = [[1, 2], [0, 2, 3], [0, 4], [1, 4], [2, 3]]
    ans = bfs(adj)
    for i in ans:
        print(i, end=" ")

#######################################################
#detectCycle
# Helper function for DFS-based cycle detection
def isCyclicUtil(adj, u, visited, recStack):
    # If the node is already in the current recursion stack, a cycle is detected
    if recStack[u]:
        return True

    # If the node is already visited and not part of the recursion stack, skip it
    if visited[u]:
        return False

    # Mark the current node as visited and add it to the recursion stack
    visited[u] = True
    recStack[u] = True

    # Recur for all the adjacent vertices
    for v in adj[u]:
        if isCyclicUtil(adj, v, visited, recStack):
            return True

    # Remove the node from the recursion stack before returning
    recStack[u] = False
    return False

# Function to build adjacency list from edge list
def constructadj(V, edges):
    adj = [[] for _ in range(V)]  # Create a list for each vertex
    for u, v in edges:
        adj[u].append(v)  # Add directed edge from u to v
    return adj

# Main function to detect cycle in the directed graph
def isCyclic(V, edges):
    adj = constructadj(V, edges)
    visited = [False] * V       # To track visited vertices
    recStack = [False] * V      # To track vertices in the current DFS path

    # Try DFS from each vertex
    for i in range(V):
        if not visited[i] and isCyclicUtil(adj, i, visited, recStack):
            return True  # Cycle found
    return False  # No cycle found


# Example usage
V = 4  # Number of vertices
edges = [[0, 1], [0, 2], [1, 2], [2, 0], [2, 3]]

# Output: True, because there is a cycle (0 → 2 → 0)
print(isCyclic(V, edges))

###########################################################
#dfsConnected.py
def dfsRec(adj, visited, s, res):
    visited[s] = True
    res.append(s)

    # Recursively visit all adjacent vertices that are not visited yet
    for i in range(len(adj)):
        if adj[s][i] == 1 and not visited[i]:
            dfsRec(adj, visited, i, res)


def DFS(adj):
    visited = [False] * len(adj)
    res = []
    dfsRec(adj, visited, 0, res)  # Start DFS from vertex 0
    return res


def add_edge(adj, s, t):
    adj[s][t] = 1
    adj[t][s] = 1  # Since it's an undirected graph


# Driver code
V = 5
adj = [[0] * V for _ in range(V)]  # Adjacency matrix

# Define the edges of the graph
edges = [(1, 2), (1, 0), (2, 0), (2, 3), (2, 4)]

# Populate the adjacency matrix with edges
for s, t in edges:
    add_edge(adj, s, t)

res = DFS(adj)  # Perform DFS
print(" ".join(map(str, res)))

#########################################################
#dfsDisconnected.py
# Create an adjacency list for the graph
from collections import defaultdict


def add_edge(adj, s, t):
    adj[s].append(t)
    adj[t].append(s)

# Recursive function for DFS traversal


def dfs_rec(adj, visited, s, res):
    # Mark the current vertex as visited
    visited[s] = True
    res.append(s)

    # Recursively visit all adjacent vertices that are not visited yet
    for i in adj[s]:
        if not visited[i]:
            dfs_rec(adj, visited, i, res)

# Main DFS function to perform DFS for the entire graph


def dfs(adj):
    visited = [False] * len(adj)
    res = []
    # Loop through all vertices to handle disconnected graph
    for i in range(len(adj)):
        if not visited[i]:
            # If vertex i has not been visited,
            # perform DFS from it
            dfs_rec(adj, visited, i, res)
    return res


V = 6
# Create an adjacency list for the graph
adj = defaultdict(list)

# Define the edges of the graph
edges = [[1, 2], [2, 0], [0, 3], [4, 5]]

# Populate the adjacency list with edges
for e in edges:
    add_edge(adj, e[0], e[1])
res = dfs(adj)

print(' '.join(map(str, res)))

#########################################################
# #kahn.py
from collections import deque

# Function to construct adjacency list from edge list
def constructadj(V, edges):
    adj = [[] for _ in range(V)]  # Initialize empty list for each vertex
    for u, v in edges:
        adj[u].append(v)          # Directed edge from u to v
    return adj

# Function to check for cycle using Kahn's Algorithm (BFS-based Topological Sort)
def isCyclic(V, edges):
    adj = constructadj(V, edges)
    in_degree = [0] * V
    queue = deque()
    visited = 0                       # Count of visited nodes

    #  Calculate in-degree of each node
    for u in range(V):
        for v in adj[u]:
            in_degree[v] += 1

    #  Enqueue nodes with in-degree 0
    for u in range(V):
        if in_degree[u] == 0:
            queue.append(u)

    #  Perform BFS (Topological Sort)
    while queue:
        u = queue.popleft()
        visited += 1

        # Decrease in-degree of adjacent nodes
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    #  If visited != V, graph has a cycle
    return visited != V


# Example usage
V = 4
edges = [[0, 1], [0, 2], [1, 2], [2, 0], [2, 3]]

# Output: true (because there is a cycle: 0 → 2 → 0)
print("true" if isCyclic(V, edges) else "false")


######################################################
#topologicalSorting.py
# Function to perform DFS and topological sorting
def topologicalSortUtil(v, adj, visited, stack):
    # Mark the current node as visited
    visited[v] = True

    # Recur for all adjacent vertices
    for i in adj[v]:
        if not visited[i]:
            topologicalSortUtil(i, adj, visited, stack)

    # Push current vertex to stack which stores the result
    stack.append(v)

# construct adj list
def constructadj(V, edges):
    adj = [[] for _ in range(V)]

    for it in edges:
        adj[it[0]].append(it[1])

    return adj

# Function to perform Topological Sort
def topologicalSort(V, edges):
    # Stack to store the result
    stack = []
    visited = [False] * V

    adj = constructadj(V, edges)
    # Call the recursive helper function to store
    # Topological Sort starting from all vertices one by one
    for i in range(V):
        if not visited[i]:
            topologicalSortUtil(i, adj, visited, stack)

    # Reverse stack to get the correct topological order
    return stack[::-1]


if __name__ == '__main__':
    # Graph represented as an adjacency list
    v = 6
    edges = [[2, 3], [3, 1], [4, 0], [4, 1], [5, 0], [5, 2]]

    ans = topologicalSort(v, edges)

    print(" ".join(map(str, ans)))

#############################################################
#bfsrottenorange.py
# Python Program to find the minimum time
# in which all oranges will get rotten
from collections import deque


# Check if i, j is within the array
# limits of row and column
def isSafe(i, j, n, m):
    return 0 <= i < n and 0 <= j < m


def orangesRotting(mat):
    n = len(mat)
    m = len(mat[0])

    # all four directions
    directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    # queue to store cell position
    q = deque()

    # find all rotten oranges
    for i in range(n):
        for j in range(m):
            if mat[i][j] == 2:
                q.append((i, j))

    # counter of elapsed time
    elapsedTime = 0

    while q:
        # increase time by 1
        elapsedTime += 1

        for _ in range(len(q)):
            i, j = q.popleft()

            # change 4-directionally connected cells
            for dir in directions:
                x = i + dir[0]
                y = j + dir[1]

                # if cell is in the matrix and
                # the orange is fresh
                if isSafe(x, y, n, m) and mat[x][y] == 1:
                    mat[x][y] = 2
                    q.append((x, y))

    # check if any fresh orange is remaining
    for i in range(n):
        for j in range(m):
            if mat[i][j] == 1:
                return -1

    return max(0, elapsedTime - 1)


if __name__ == "__main__":
    mat = [[2, 1, 0, 2, 1],
           [1, 0, 1, 2, 1],
           [1, 0, 0, 2, 1]]

    print(orangesRotting(mat))

##################################################
#dfsRotternOrange.py
# Python Program to find the minimum time
# in which all oranges will get rotten

def is_safe(i, j, n, m):
    return 0 <= i < n and 0 <= j < m

# function to perform dfs and find fresh orange


def dfs(mat, i, j, time):
    n = len(mat)
    m = len(mat[0])

    # update minimum time
    mat[i][j] = time

    # all four directions
    directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]

    # change 4-directionally connected cells
    for dir in directions:
        x = i + dir[0]
        y = j + dir[1]

        # if cell is in the matrix and
        # the orange is fresh
        if is_safe(x, y, n, m) and (mat[x][y] == 1 or mat[x][y] > time + 1):
            dfs(mat, x, y, time + 1)


def oranges_rotting(mat):
    n = len(mat)
    m = len(mat[0])

    # counter of elapsed time
    elapsed_time = 0

    # iterate through all the cells
    for i in range(n):
        for j in range(m):

            # if orange is initially rotten
            if mat[i][j] == 2:
                dfs(mat, i, j, 2)

    # iterate through all the cells
    for i in range(n):
        for j in range(m):

            # if orange is fresh
            if mat[i][j] == 1:
                return -1

            # update the maximum time
            elapsed_time = max(elapsed_time, mat[i][j] - 2)

    return elapsed_time


if __name__ == "__main__":
    mat = [[2, 1, 0, 2, 1],
          [1, 0, 1, 2, 1],
          [1, 0, 0, 2, 1]]

    print(oranges_rotting(mat))
    ####################################################
    #dijkstra.py
    import heapq
    import sys


    # Function to construct adjacency
    def constructAdj(edges, V):

        # adj[u] = list of [v, wt]
        adj = [[] for _ in range(V)]

        for edge in edges:
            u, v, wt = edge
            adj[u].append([v, wt])
            adj[v].append([u, wt])

        return adj


    # Returns shortest distances from src to all other vertices
    def dijkstra(V, edges, src):
        # Create adjacency list
        adj = constructAdj(edges, V)

        # Create a priority queue to store vertices that
        # are being preprocessed.
        pq = []

        # Create a list for distances and initialize all
        # distances as infinite
        dist = [sys.maxsize] * V

        # Insert source itself in priority queue and initialize
        # its distance as 0.
        heapq.heappush(pq, [0, src])
        dist[src] = 0

        # Looping till priority queue becomes empty (or all
        # distances are not finalized)
        while pq:
            # The first vertex in pair is the minimum distance
            # vertex, extract it from priority queue.
            u = heapq.heappop(pq)[1]

            # Get all adjacent of u.
            for x in adj[u]:
                # Get vertex label and weight of current
                # adjacent of u.
                v, weight = x[0], x[1]

                # If there is shorter path to v through u.
                if dist[v] > dist[u] + weight:
                    # Updating distance of v
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, [dist[v], v])

        # Return the shortest distance array
        return dist


    # Driver program to test methods of graph class
    if __name__ == "__main__":
        V = 5
        src = 0

        # edge list format: {u, v, weight}
        edges = [[0, 1, 4], [0, 2, 8], [1, 4, 6], [2, 3, 2], [3, 4, 10]];

        result = dijkstra(V, edges, src)

        # Print shortest distances in one line
        print(' '.join(map(str, result)))

###########################################################
#numberofislands.py
def isSafe(grid, r, c, visited):
    row = len(grid)
    col = len(grid[0])

    return (0 <= r < row) and (0 <= c < col) and (grid[r][c] == 'L' and not visited[r][c])


def dfs(grid, r, c, visited):
    rNbr = [-1, -1, -1, 0, 0, 1, 1, 1]
    cNbr = [-1, 0, 1, -1, 1, -1, 0, 1]

    # Mark this cell as visited
    visited[r][c] = True

    # Recur for all connected neighbours
    for k in range(8):
        newR, newC = r + rNbr[k], c + cNbr[k]
        if isSafe(grid, newR, newC, visited):
            dfs(grid, newR, newC, visited)


def countIslands(grid):
    row = len(grid)
    col = len(grid[0])

    visited = [[False for _ in range(col)] for _ in range(row)]

    count = 0
    for r in range(row):
        for c in range(col):

            # If a cell with value 'L' (land) is not visited yet,
            # then a new island is found
            if grid[r][c] == 'L' and not visited[r][c]:
                # Visit all cells in this island.
                dfs(grid, r, c, visited)

                # increment the island count
                count += 1
    return count


if __name__ == "__main__":
    grid = [
        ['L', 'L', 'W', 'W', 'W'],
        ['W', 'L', 'W', 'W', 'L'],
        ['L', 'W', 'W', 'L', 'L'],
        ['W', 'W', 'W', 'W', 'W'],
        ['L', 'W', 'L', 'L', 'W']
    ]

    print(countIslands(grid))

#################################################
#shortestpathindag.py
# Python program to find single source shortest paths
# for Directed Acyclic Graphs Complexity :O(V+E)
from collections import defaultdict

# Graph is represented using adjacency list. Every
# node of adjacency list contains vertex number of
# the vertex to which edge connects. It also contains
# weight of the edge
class Graph:
	def __init__(self,vertices):

		self.V = vertices # No. of vertices

		# dictionary containing adjacency List
		self.graph = defaultdict(list)

	# function to add an edge to graph
	def addEdge(self,u,v,w):
		self.graph[u].append((v,w))


	# A recursive function used by shortestPath
	def topologicalSortUtil(self,v,visited,stack):

		# Mark the current node as visited.
		visited[v] = True

		# Recur for all the vertices adjacent to this vertex
		if v in self.graph.keys():
			for node,weight in self.graph[v]:
				if visited[node] == False:
					self.topologicalSortUtil(node,visited,stack)

		# Push current vertex to stack which stores topological sort
		stack.append(v)


	''' The function to find shortest paths from given vertex.
		It uses recursive topologicalSortUtil() to get topological
		sorting of given graph.'''
	def shortestPath(self, s):

		# Mark all the vertices as not visited
		visited = [False]*self.V
		stack =[]

		# Call the recursive helper function to store Topological
		# Sort starting from source vertices
		for i in range(self.V):
			if visited[i] == False:
				self.topologicalSortUtil(i,visited,stack)

		# Initialize distances to all vertices as infinite and
		# distance to source as 0
		dist = [float("Inf")] * (self.V)
		dist[s] = 0

		# Process vertices in topological order
		while stack:

			# Get the next vertex from topological order
			i = stack.pop()

			# Update distances of all adjacent vertices
			for node,weight in self.graph[i]:
				if dist[node] > dist[i] + weight:
					dist[node] = dist[i] + weight

		# Print the calculated shortest distances
		for i in range(self.V):
			print (("%d" %dist[i]) if dist[i] != float("Inf") else "Inf" ,end=" ")


g = Graph(6)
g.addEdge(0, 1, 5)
g.addEdge(0, 2, 3)
g.addEdge(1, 3, 6)
g.addEdge(1, 2, 2)
g.addEdge(2, 4, 4)
g.addEdge(2, 5, 2)
g.addEdge(2, 3, 7)
g.addEdge(3, 4, -1)
g.addEdge(4, 5, -2)

# source = 1
s = 1

print ("Following are shortest distances from source %d " % s)
g.shortestPath(s)
#############################################################
#bellmanFord.py
def bellmanFord(V, edges, src):
    # Initially distance from source to all other vertices
    # is not known(Infinite) e.g. 1e8.
    dist = [100000000] * V
    dist[src] = 0

    # Relaxation of all the edges V times, not (V - 1) as we
    # need one additional relaxation to detect negative cycle
    for i in range(V):
        for edge in edges:
            u, v, wt = edge
            if dist[u] != 100000000 and dist[u] + wt < dist[v]:

                # If this is the Vth relaxation, then there
                # is a negative cycle
                if i == V - 1:
                    return [-1]

                # Update shortest distance to node v
                dist[v] = dist[u] + wt
    return dist


if __name__ == '__main__':
    V = 5
    edges = [[1, 3, 2], [4, 3, -1], [2, 4, 1], [1, 2, 1], [0, 1, 5]]

    src = 0
    ans = bellmanFord(V, edges, src)
    print(' '.join(map(str, ans)))

############################################################
#cheapestFlightWithinKstops.py
from typing import List


class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float("inf")] * n
        prices[src] = 0

        for i in range(k + 1):
            tmpPrices = prices.copy()
            for s, d, p in flights:
                if prices[s] == float("inf"):
                    continue
                if prices[s] + p < tmpPrices[d]:
                    tmpPrices[d] = prices[s] + p
            prices = tmpPrices

        return -1 if prices[dst] == float("inf") else prices[dst]


# Driver Code
if __name__ == "__main__":
    solution = Solution()

    # example 1
    n = 4
    flights = [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]]
    src = 0
    dst = 3
    k = 1

    result = solution.findCheapestPrice(n, flights, src, dst, k)
    print(f"The cheapest flight within {k} stops is: {result}")

####################################################
#courseSchedule.py
from typing import List

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Map each course to its prerequisite list
        preMap = {i: [] for i in range(numCourses)}
        for crs, pre in prerequisites:
            preMap[crs].append(pre)

        # visitSet = all courses along the current DFS path
        visitSet = set()

        def dfs(crs):
            if crs in visitSet:
                return False
            if preMap[crs] == []:
                return True

            visitSet.add(crs)
            for pre in preMap[crs]:
                if not dfs(pre):
                    return False
            visitSet.remove(crs)
            preMap[crs] = []
            return True

        for crs in range(numCourses):
            if not dfs(crs):
                return False
        return True

# -------------------------------
# Driver code to test the function
# -------------------------------

if __name__ == "__main__":
    sol = Solution()

    # Example 1
    numCourses1 = 2
    prerequisites1 = [[1, 0]]
    print("Example 1 - Can finish:", sol.canFinish(numCourses1, prerequisites1))  # Expected: True

    # Example 2
    numCourses2 = 2
    prerequisites2 = [[1, 0], [0, 1]]
    print("Example 2 - Can finish:", sol.canFinish(numCourses2, prerequisites2))  # Expected: False

    # Example 3
    numCourses3 = 4
    prerequisites3 = [[1, 0], [2, 1], [3, 2]]
    print("Example 3 - Can finish:", sol.canFinish(numCourses3, prerequisites3))  # Expected: True

    # Example 4 (Cycle)
    numCourses4 = 4
    prerequisites4 = [[1, 0], [2, 1], [0, 2], [3, 2]]
    print("Example 4 - Can finish:", sol.canFinish(numCourses4, prerequisites4))  # Expected: False

###################################################
#courseSchedulell.py
from collections import deque

def findOrder(n, prerequisites):
    adj = [[] for _ in range(n)]
    inDegree = [0] * n

    for dest, src in prerequisites:
        adj[src].append(dest)
        inDegree[dest] += 1

    q = deque([i for i in range(n) if inDegree[i] == 0])
    order = []

    while q:
        current = q.popleft()
        order.append(current)

        for neighbor in adj[current]:
            inDegree[neighbor] -= 1
            if inDegree[neighbor] == 0:
                q.append(neighbor)

    return order if len(order) == n else []

# Example
if __name__ == "__main__":
    n = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print("Course order:", findOrder(n, prerequisites))

###########################################################
#networkdelaytime.py
import heapq
import collections
from typing import List

class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        edges = collections.defaultdict(list)
        for u, v, w in times:
            edges[u].append((v, w))

        minHeap = [(0, k)]
        visit = set()
        t = 0

        while minHeap:
            w1, n1 = heapq.heappop(minHeap)
            if n1 in visit:
                continue
            visit.add(n1)
            t = max(t, w1)

            for n2, w2 in edges[n1]:
                if n2 not in visit:
                    heapq.heappush(minHeap, (w1 + w2, n2))

        return t if len(visit) == n else -1

# -------------------------------
# Driver code to test the function
# -------------------------------
if __name__ == "__main__":
    sol = Solution()

    # Example 1
    times1 = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    n1 = 4
    k1 = 2
    print("Example 1 - Network delay time:", sol.networkDelayTime(times1, n1, k1))
    # Expected: 2

    # Example 2
    times2 = [[1, 2, 1]]
    n2 = 2
    k2 = 1
    print("Example 2 - Network delay time:", sol.networkDelayTime(times2, n2, k2))
    # Expected: 1

    # Example 3 (Disconnected)
    times3 = [[1, 2, 1]]
    n3 = 2
    k3 = 2
    print("Example 3 - Network delay time:", sol.networkDelayTime(times3, n3, k3))
    # Expected: -1


################################################
#climbingstairsbottomup.py
# Python program to count number of ways
# to reach nth stair using Tabulation

def countWays(n):
    dp = [0] * (n + 1)

    # Base cases
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2];

    return dp[n]


n = 4
print(countWays(n))

##############################################
#climbingStairsRec.py
# Python program to count number of ways to reach
# nth stair using recursion

def countWays(n):

    # Base cases: If there are 0 or 1 stairs,
    # there is only one way to reach the top.
    if n == 0 or n == 1:
        return 1

    return countWays(n - 1) + countWays(n - 2)

n = 4
print(countWays(n))
################################################
#climbingStairsSpaceOptimised.py
# Python program to count number of ways to
# reach nth stair using Space Optimized DP

def countWays(n):
    # variable prev1, prev2 - to store the
    # values of last and second last states
    prev1 = 1
    prev2 = 1

    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    # In last iteration final value
    # of curr is stored in prev.
    return prev1


n = 4
print(countWays(n))

######################################################
#climbingstairsTopDown.py
# Python program to count number of ways to reach nth stair
# using memoization

def countWaysRec(n, memo):
    # Base cases
    if n == 0 or n == 1:
        return 1

    # if the result for this subproblem is
    # already computed then return it
    if memo[n] != -1:
        return memo[n]

    memo[n] = countWaysRec(n - 1, memo) + countWaysRec(n - 2, memo)
    return memo[n]


def countWays(n):
    # Memoization array to store the results
    memo = [-1] * (n + 1)
    return countWaysRec(n, memo)


if __name__ == "__main__":
    n = 4
    print(countWays(n))
###################################################
#fibonacciExpected.py
# Function to calculate the nth Fibonacci number using memoization
def nth_fibonacci_util(n, memo):

    # Base case: if n is 0 or 1, return n
    if n <= 1:
        return n

    # Check if the result is already in the memo table
    if memo[n] != -1:
        return memo[n]

    # Recursive case: calculate Fibonacci number
    # and store it in memo
    memo[n] = nth_fibonacci_util(n - 1, memo) + nth_fibonacci_util(n - 2, memo)

    return memo[n]


# Wrapper function that handles both initialization
# and Fibonacci calculation
def nth_fibonacci(n):

    # Create a memoization table and initialize with -1
    memo = [-1] * (n + 1)

    # Call the utility function
    return nth_fibonacci_util(n, memo)


if __name__ == "__main__":
    n = 5
    result = nth_fibonacci(n)
    print(result)
####################################################
#fibonacciExpectedll.py
def nth_fibonacci(n):
    # Handle the edge cases
    if n <= 1:
        return n

    # Create a list to store Fibonacci numbers
    dp = [0] * (n + 1)

    # Initialize the first two Fibonacci numbers
    dp[0] = 0
    dp[1] = 1

    # Fill the list iteratively
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    # Return the nth Fibonacci number
    return dp[n]


n = 5
result = nth_fibonacci(n)
print(result)
######################################################
#fibonacciNaive.py
def nth_fibonacci(n):
    # Base case: if n is 0 or 1, return n
    if n <= 1:
        return n

    # Recursive case: sum of the two preceding Fibonacci numbers
    return nth_fibonacci(n - 1) + nth_fibonacci(n - 2)


n = 5
result = nth_fibonacci(n)
print(result)

####################################################
#fibSpaceOpt.py
def nth_fibonacci(n):
    if n <= 1:
        return n

    # To store the curr Fibonacci number
    curr = 0

    # To store the previous Fibonacci numbers
    prev1 = 1
    prev2 = 0

    # Loop to calculate Fibonacci numbers from 2 to n
    for i in range(2, n + 1):
        # Calculate the curr Fibonacci number
        curr = prev1 + prev2

        # Update prev2 to the last Fibonacci number
        prev2 = prev1

        # Update prev1 to the curr Fibonacci number
        prev1 = curr

    return curr


n = 5
result = nth_fibonacci(n)
print(result)
#################################################
#houseRobberBottomUp.py
# Python Program to solve House Robber Problem using Tabulation

def maxLoot(hval):
    n = len(hval)

    # Create a dp array to store the maximum loot at each house
    dp = [0] * (n + 1)

    # Base cases
    dp[0] = 0
    dp[1] = hval[0]

    # Fill the dp array using the bottom-up approach
    for i in range(2, n + 1):
        dp[i] = max(hval[i - 1] + dp[i - 2], dp[i - 1])

    return dp[n]


hval = [6, 7, 1, 3, 8, 2, 4]
print(maxLoot(hval))

##########################################
#houseRobberNaive.py
# Python Program to solve House Robber Problem using Recursion

# Calculate the maximum stolen value recursively
def maxLootRec(hval, n):
    # If no houses are left, return 0.
    if n <= 0:
        return 0

    # If only 1 house is left, rob it.
    if n == 1:
        return hval[0]

    # Two Choices: Rob the nth house and do not rob the nth house
    pick = hval[n - 1] + maxLootRec(hval, n - 2)
    notPick = maxLootRec(hval, n - 1)

    # Return the max of two choices
    return max(pick, notPick)


# Function to calculate the maximum stolen value
def maxLoot(hval):
    n = len(hval)

    # Call the recursive function for n houses
    return maxLootRec(hval, n)


if __name__ == "__main__":
    hval = [6, 7, 1, 3, 8, 2, 4]
    print(maxLoot(hval))

###################################################
#houseRobberSpaceOpt.py

# Python Program to solve House Robber Problem using
# Space Optimized Tabulation

# Function to calculate the maximum stolen value
def maxLoot(hval):
    n = len(hval)

    if n == 0:
        return 0
    if n == 1:
        return hval[0]

    # Set previous 2 values
    secondLast = 0
    last = hval[0]

    # Compute current value using previous two values
    # The final current value would be our result
    res = 0
    for i in range(1, n):
        res = max(hval[i] + secondLast, last)
        secondLast = last
        last = res

    return res

hval = [6, 7, 1, 3, 8, 2, 4]
print(maxLoot(hval))
#################################################
#houserobbermemorization.py
# Python Program to solve House Robber Problem using Memoization

def maxLootRec(hval, n, memo):
    if n <= 0:
        return 0
    if n == 1:
        return hval[0]

    # Check if the result is already computed
    if memo[n] != -1:
        return memo[n]

    pick = hval[n - 1] + maxLootRec(hval, n - 2, memo)
    notPick = maxLootRec(hval, n - 1, memo)

    # Store the max of two choices in the memo array and return it
    memo[n] = max(pick, notPick)
    return memo[n]


def maxLoot(hval):
    n = len(hval)

    # Initialize memo array with -1
    memo = [-1] * (n + 1)
    return maxLootRec(hval, n, memo)


if __name__ == "__main__":
    hval = [6, 7, 1, 3, 8, 2, 4]
    print(maxLoot(hval))
###############################################
#textJustification.py
from typing import List

class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res = []
        i = 0
        width = 0
        cur_line = []

        while i < len(words):
            cur_word = words[i]

            if width + len(cur_word) <= maxWidth:
                cur_line.append(cur_word)
                width += len(cur_word) + 1
                i += 1
            else:
                spaces = maxWidth - width + len(cur_line)
                added = 0
                j = 0

                while added < spaces:
                    if j >= len(cur_line) - 1:
                        j = 0
                    cur_line[j] += " "
                    added += 1
                    j += 1

                res.append("".join(cur_line))
                cur_line = []
                width = 0

        for word in range(len(cur_line) - 1):
            cur_line[word] += " "

        cur_line[-1] += " " * (maxWidth - width + 1)
        res.append("".join(cur_line))

        return res


def run_examples():
    solution = Solution()

    examples = [
        {
            "words": ["This", "is", "an", "example", "of", "text", "justification."],
            "maxWidth": 16
        },
        {
            "words": ["What","must","be","acknowledgment","shall","be"],
            "maxWidth": 16
        },
        {
            "words": ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"],
            "maxWidth": 20
        }
    ]

    for idx, example in enumerate(examples, 1):
        print(f"Example {idx}:")
        output = solution.fullJustify(example["words"], example["maxWidth"])
        for line in output:
            print(f'"{line}"')
        print("-" * 40)


run_examples()
################################################
#wordBreakBottomUp.py
# Python program to implement word break
def wordBreak(s, dictionary):
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    # Traverse through the given string
    for i in range(1, n + 1):

        # Traverse through the dictionary words
        for w in dictionary:

            # Check if current word is present
            # the prefix before the word is also
            # breakable
            start = i - len(w)
            if start >= 0 and dp[start] and s[start:start + len(w)] == w:
                dp[i] = True
                break
    return 1 if dp[n] else 0


if __name__ == '__main__':
    s = "ilike"

    dictionary = ["i", "like", "gfg"]

    print("true" if wordBreak(s, dictionary) else "false")
#######################################################
#wordBreakMemo.py
def wordBreakRec(ind, s, dict, dp):
    if ind >= len(s):
        return True
    if dp[ind] != -1:
        return dp[ind] == 1
    possible = False
    for temp in dict:
        if len(temp) > len(s) - ind:
            continue
        if s[ind:ind+len(temp)] == temp:
            possible |= wordBreakRec(ind + len(temp), s, dict, dp)
    dp[ind] = 1 if possible else 0
    return possible

def word_break(s, dict):
    n = len(s)
    dp = [-1] * (n + 1)
    return wordBreakRec(0, s, dict, dp)

s = "ilike"
dict = ["i", "like", "gfg"]
print("true" if word_break(s, dict) else "false")
########################################################
#wordBreakNaive.py
def wordBreakRec(i, s, dictionary):
    # If end of string is reached,
    # return true.
    if i == len(s):
        return 1

    n = len(s)
    prefix = ""

    # Try every prefix
    for j in range(i, n):
        prefix += s[j]

        # if the prefix s[i..j] is a dictionary word
        # and rest of the string can also be broken into
        # valid words, return true
        if prefix in dictionary and wordBreakRec(j + 1, s, dictionary) == 1:
            return 1
    return 0


def wordBreak(s, dictionary):
    return wordBreakRec(0, s, dictionary)


if __name__ == "__main__":
    s = "ilike"

    dictionary = {"i", "like", "gfg"}

    print("true" if wordBreak(s, dictionary) else "false")

######################################################
#editDistanceBU.py
# Python program to find minimum number
# of operations to convert s1 to s2

# Function to find the minimum number
# of operations to convert s1 to s2
def editDistance(s1, s2):
    m = len(s1)
    n = len(s2)

    # Create a table to store results of subproblems
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the known entries in dp[][]
    # If one string is empty, then answer
    # is length of the other string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the rest of dp[][]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])

    return dp[m][n]

if __name__ == "__main__":
    s1 = "abcd"
    s2 = "bcfe"

    print(editDistance(s1, s2))

##########################################
#editDistanceNaive.py
# A Naive recursive Python program to find minimum number
# of operations to convert s1 to s2.

# Recursive function to find number of operations
# needed to convert s1 into s2.
def editDistRec(s1, s2, m, n):

    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

    # If last characters of two strings are same, nothing
    # much to do. Get the count for
    # remaining strings.
    if s1[m - 1] == s2[n - 1]:
        return editDistRec(s1, s2, m - 1, n - 1)

    # If last characters are not same, consider all three
    # operations on last character of first string,
    # recursively compute minimum cost for all three
    # operations and take minimum of three values.
    return 1 + min(editDistRec(s1, s2, m, n - 1),
                   editDistRec(s1, s2, m - 1, n),
                   editDistRec(s1, s2, m - 1, n - 1))

# Wrapper function to initiate
# the recursive calculation
def editDistance(s1, s2):
    return editDistRec(s1, s2, len(s1), len(s2))


if __name__ == "__main__":
    s1 = "abcd"
    s2 = "bcfe"

    print(editDistance(s1, s2))
############################################################
#editDistanceTD.py
# Python program to find minimum number
# of operations to convert s1 to s2

# Recursive function to find number of operations
# needed to convert s1 into s2.
def editDistRec(s1, s2, m, n, memo):
    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

    # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

    # If value is memoized
    if memo[m][n] != -1:
        return memo[m][n]

    # If last characters of two strings are same, nothing
    # much to do. Get the count for
    # remaining strings.
    if s1[m - 1] == s2[n - 1]:
        memo[m][n] = editDistRec(s1, s2, m - 1, n - 1, memo)
        return memo[m][n]

    # If last characters are not same, consider all three
    # operations on last character of first string,
    # recursively compute minimum cost for all three
    # operations and take minimum of three values.
    memo[m][n] = 1 + min(
        editDistRec(s1, s2, m, n - 1, memo),
        editDistRec(s1, s2, m - 1, n, memo),
        editDistRec(s1, s2, m - 1, n - 1, memo)
    )
    return memo[m][n]


# Wrapper function to initiate the recursive calculation
def editDistance(s1, s2):
    m, n = len(s1), len(s2)
    memo = [[-1 for _ in range(n + 1)] for _ in range(m + 1)]
    return editDistRec(s1, s2, m, n, memo)


if __name__ == "__main__":
    s1 = "abcd"
    s2 = "bcfe"
    print(editDistance(s1, s2))


##################################################
#knapsackBU.py
def knapsack(W, val, wt):
    n = len(wt)
    dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]

    # Build table dp[][] in bottom-up manner
    for i in range(n + 1):
        for j in range(W + 1):

            # If there is no item or the knapsack's capacity is 0
            if i == 0 or j == 0:
                dp[i][j] = 0
            else:
                pick = 0

                # Pick ith item if it does not exceed the capacity of knapsack
                if wt[i - 1] <= j:
                    pick = val[i - 1] + dp[i - 1][j - wt[i - 1]]

                # Don't pick the ith item
                notPick = dp[i - 1][j]

                dp[i][j] = max(pick, notPick)

    return dp[n][W]


if __name__ == "__main__":
    val = [1, 2, 3]
    wt = [4, 5, 1]
    W = 4

    print(knapsack(W, val, wt))

######################################################
#knapsackRec.py
# Returns the maximum value that
# can be put in a knapsack of capacity W
def knapsackRec(W, val, wt, n):
    # Base Case
    if n == 0 or W == 0:
        return 0

    pick = 0

    # Pick nth item if it does not exceed the capacity of knapsack
    if wt[n - 1] <= W:
        pick = val[n - 1] + knapsackRec(W - wt[n - 1], val, wt, n - 1)

    # Don't pick the nth item
    notPick = knapsackRec(W, val, wt, n - 1)

    return max(pick, notPick)


def knapsack(W, val, wt):
    n = len(val)
    return knapsackRec(W, val, wt, n)


if __name__ == "__main__":
    val = [1, 2, 3]
    wt = [4, 5, 1]
    W = 4

    print(knapsack(W, val, wt))

######################################################
#knapsackTD.py
# Returns the maximum value that
# can be put in a knapsack of capacity W
def knapsackRec(W, val, wt, n, memo):

    # Base Case
    if n == 0 or W == 0:
        return 0

    # Check if we have previously calculated the same subproblem
    if memo[n][W] != -1:
        return memo[n][W]

    pick = 0

    # Pick nth item if it does not exceed the capacity of knapsack
    if wt[n - 1] <= W:
        pick = val[n - 1] + knapsackRec(W - wt[n - 1], val, wt, n - 1, memo)

    # Don't pick the nth item
    notPick = knapsackRec(W, val, wt, n - 1, memo)

    # Store the result in memo[n][W] and return it
    memo[n][W] = max(pick, notPick)
    return memo[n][W]

def knapsack(W, val, wt):
    n = len(val)

    # Memoization table to store the results
    memo = [[-1] * (W + 1) for _ in range(n + 1)]

    return knapsackRec(W, val, wt, n, memo)

if __name__ == "__main__":
    val = [1, 2, 3]
    wt = [4, 5, 1]
    W = 4

    print(knapsack(W, val, wt))

#######################################################
#optimalParametherizationBU.py
# Python program to find Optimal parenthesization
# using Tabulation

def matrixChainOrder(arr):
    n = len(arr)

    # dp[i][j] stores a pair: matrix order,
    # minimum cost
    dp = [[("", 0) for i in range(n)] for i in range(n)]

    # Base Case: Initializing diagonal of the dp
    # Cost for multiplying a single matrix is zero
    for i in range(n):
        temp = ""

        # Label the matrices as A, B, C, ...
        temp += chr(ord('A') + i)

        # No cost for multiplying a single matrix
        dp[i][i] = (temp, 0)

    # Fill the DP table for chain lengths
    # greater than 1
    for length in range(2, n):
        for i in range(n - length):
            j = i + length - 1
            cost = float("inf")
            str = ""

            # Try all possible split points k
            # between i and j
            for k in range(i + 1, j + 1):

                # Calculate the cost of multiplying
                # matrices from i to k and from k to j
                currCost = (
                    dp[i][k - 1][1] + dp[k][j][1]
                    + arr[i] * arr[k] * arr[j + 1]
                )

                # Update if we find a lower cost
                if currCost < cost:
                    cost = currCost
                    str = "(" + dp[i][k - 1][0] + dp[k][j][0] + ")"

            dp[i][j] = (str, cost)

    # Return the optimal matrix order for
    # the entire chain
    return dp[0][n - 2][0]

arr = [40, 20, 30, 10, 30]
print(matrixChainOrder(arr))

#####################################################
#optimalParametherizationRec.py
# Python program to find Optimal parenthesization
# using Recursion

def matrixChainOrderRec(arr, i, j):
    # If there is only one matrix
    if i == j:
        temp = chr(ord('A') + i)
        return (temp, 0)

    res = float('inf')
    str = ""
    # Try all possible split points k between i and j
    for k in range(i + 1, j + 1):
        left = matrixChainOrderRec(arr, i, k - 1)
        right = matrixChainOrderRec(arr, k, j)
        # Calculate the cost of multiplying
        # matrices from i to k and from k to j
        currCost = left[1] + right[1] + arr[i] * arr[k] * arr[j + 1]
        # Update if we find a lower cost
        if res > currCost:
            res = currCost
            str = "(" + left[0] + right[0] + ")"

    return (str, res)

def matrixChainOrder(arr):
    n = len(arr)
    return matrixChainOrderRec(arr, 0, n - 2)[0]

arr = [40, 20, 30, 10, 30]
print(matrixChainOrder(arr))

################################################
#optimalParametherizationTD.py
# Python program to find Optimal parenthesization
# using Memoization

def matrixChainOrderRec(arr, i, j, memo):
    # If there is only one matrix
    if i == j:
        temp = chr(ord('A') + i)
        return (temp, 0)

    # if the result for this subproblem is
    # already computed then return it
    if memo[i][j][1] != -1:
        return memo[i][j]

    res = float('inf')
    str = ""

    # Try all possible split points k between i and j
    for k in range(i + 1, j + 1):
        left = matrixChainOrderRec(arr, i, k - 1, memo)
        right = matrixChainOrderRec(arr, k, j, memo)

        # Calculate the cost of multiplying
        # matrices from i to k and from k to j
        curr = left[1] + right[1] + arr[i] * arr[k] * arr[j + 1]

        # Update if we find a lower cost
        if res > curr:
            res = curr
            str = "(" + left[0] + right[0] + ")"

    # Return minimum cost and matrix
    # multiplication order
    memo[i][j] = (str, res)
    return memo[i][j]


def matrixChainOrder(arr):
    n = len(arr)

    # Memoization array to store the results
    memo = [[("", -1) for i in range(n)] for i in range(n)]

    return matrixChainOrderRec(arr, 0, n - 2, memo)[0]


arr = [40, 20, 30, 10, 30]
print(matrixChainOrder(arr))
###################################################
#longestCommonSubsequenceBU.py
def lcs(S1, S2):
    m = len(S1)
    n = len(S2)

    # Initializing a matrix of size (m+1)*(n+1)
    dp = [[0] * (n + 1) for x in range(m + 1)]

    # Building dp[m+1][n+1] in bottom-up fashion
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if S1[i - 1] == S2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j],
                               dp[i][j - 1])

    # dp[m][n] contains length of LCS for S1[0..m-1]
    # and S2[0..n-1]
    return dp[m][n]


if __name__ == "__main__":
    S1 = "AGGTAB"
    S2 = "GXTXAYB"
    print(lcs(S1, S2))

##########################################################
#longestCommonSubsequenceRec.py
# A Naive recursive implementation of LCS problem

# Returns length of LCS for s1[0..m-1], s2[0..n-1]
def lcsRec(s1, s2, m, n):
    # Base case: If either string is empty, the length of LCS is 0
    if m == 0 or n == 0:
        return 0

    # If the last characters of both substrings match
    if s1[m - 1] == s2[n - 1]:

        # Include this character in LCS and recur for remaining substrings
        return 1 + lcsRec(s1, s2, m - 1, n - 1)

    else:
        # If the last characters do not match
        # Recur for two cases:
        # 1. Exclude the last character of S1
        # 2. Exclude the last character of S2
        # Take the maximum of these two recursive calls
        return max(lcsRec(s1, s2, m, n - 1), lcsRec(s1, s2, m - 1, n))


def lcs(s1, s2):
    m = len(s1)
    n = len(s2)
    return lcsRec(s1, s2, m, n)


if __name__ == "__main__":
    s1 = "AGGTAB"
    s2 = "GXTXAYB"
    print(lcs(s1, s2))

################################################
#longestCommonSubsequenceTD.py
def lcsRec(s1, s2, m, n, memo):
    # Base Case
    if m == 0 or n == 0:
        return 0

    # Already exists in the memo table
    if memo[m][n] != -1:
        return memo[m][n]

    # Match
    if s1[m - 1] == s2[n - 1]:
        memo[m][n] = 1 + lcsRec(s1, s2, m - 1, n - 1, memo)
        return memo[m][n]

    # Do not match
    memo[m][n] = max(lcsRec(s1, s2, m, n - 1, memo),
                     lcsRec(s1, s2, m - 1, n, memo))
    return memo[m][n]


def lcs(s1, s2):
    m = len(s1)
    n = len(s2)
    memo = [[-1 for _ in range(n + 1)] for _ in range(m + 1)]
    return lcsRec(s1, s2, m, n, memo)


if __name__ == "__main__":
    s1 = "AGGTAB"
    s2 = "GXTXAYB"
    print(lcs(s1, s2))

####################################################
#longestPalindromicSubstringBU.py
def longestPalindrome(s):
    n = len(s)
    if n == 0:
        return ""

    # Create a 2D table to store palindrome truth values
    table = [[False] * n for _ in range(n)]

    start = 0  # Starting index of the longest palindrome
    max_len = 1  # At least every single character is a palindrome

    # Every single character is a palindrome
    for i in range(n):
        table[i][i] = True

    # Check for sub-strings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            table[i][i + 1] = True
            start = i
            max_len = 2

    # Check for lengths greater than 2
    for length in range(3, n + 1):  # length of the substring
        for i in range(n - length + 1):
            j = i + length - 1  # Ending index

            # Check if substring s[i+1..j-1] is a palindrome
            if s[i] == s[j] and table[i + 1][j - 1]:
                table[i][j] = True
                if length > max_len:
                    start = i
                    max_len = length

    return s[start:start + max_len]


# ----------------------------
# Driver Code to test examples
# ----------------------------

test_cases = [
    {"input": "babad", "expected_outputs": ["bab", "aba"]},
    {"input": "cbbd", "expected_outputs": ["bb"]}
]

for idx, test in enumerate(test_cases, 1):
    result = longestPalindrome(test["input"])
    if result in test["expected_outputs"]:
        print(f"Example {idx} Passed: Output = \"{result}\"")
    else:
        print(f"Example {idx} Failed: Output = \"{result}\", Expected one of {test['expected_outputs']}")

######################################################
#longestOalindromicSubstringRec.py
def checkPal(str, low, high):
    while low < high:
        if str[low] != str[high]:
            return False
        low += 1
        high -= 1
    return True

def longestPalindrome(s):
    n = len(s)
    maxLen = 1
    start = 0

    for i in range(n):
        for j in range(i, n):
            if checkPal(s, i, j) and (j - i + 1) > maxLen:
                start = i
                maxLen = j - i + 1

    return s[start:start + maxLen]

# ----------------------------
# Driver Code to test examples
# ----------------------------

test_cases = [
    {"input": "babad", "expected_outputs": ["bab", "aba"]},
    {"input": "cbbd", "expected_outputs": ["bb"]}
]

for idx, test in enumerate(test_cases, 1):
    result = longestPalindrome(test["input"])
    if result in test["expected_outputs"]:
        print(f"Example {idx} Passed: Output = \"{result}\"")
    else:
        print(f"Example {idx} Failed: Output = \"{result}\", Expected one of {test['expected_outputs']}")

################################################
#longestPalindromicSubstringTD.py
def longestPalindrome(s):
    n = len(s)
    memo = {}  # (i, j): is_palindrome

    max_len = 0
    start = 0

    def is_palindrome(i, j):
        if i >= j:
            return True
        if (i, j) in memo:
            return memo[(i, j)]
        if s[i] == s[j] and is_palindrome(i + 1, j - 1):
            memo[(i, j)] = True
        else:
            memo[(i, j)] = False
        return memo[(i, j)]

    for i in range(n):
        for j in range(i, n):
            if is_palindrome(i, j) and (j - i + 1) > max_len:
                start = i
                max_len = j - i + 1

    return s[start:start + max_len]


# ----------------------------
# Driver Code to test examples
# ----------------------------

test_cases = [
    {"input": "babad", "expected_outputs": ["bab", "aba"]},
    {"input": "cbbd", "expected_outputs": ["bb"]},
    {"input": "forgeeksskeegfor", "expected_outputs": ["geeksskeeg"]},
]

for idx, test in enumerate(test_cases, 1):
    result = longestPalindrome(test["input"])
    if result in test["expected_outputs"]:
        print(f"Example {idx} Passed: Output = \"{result}\"")
    else:
        print(f"Example {idx} Failed: Output = \"{result}\", Expected one of {test['expected_outputs']}")
#####################################################
#uniquePaths.py
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # dp[i][j] := the number of unique paths from (0, 0) to (i, j)
        dp = [[1] * n for _ in range(m)]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[-1][-1]


def test_unique_paths():
    solution = Solution()

    test_cases = [
        {"m": 3, "n": 7, "expected": 28},
        {"m": 3, "n": 2, "expected": 3},
        {"m": 7, "n": 3, "expected": 28},
        {"m": 3, "n": 3, "expected": 6},
        {"m": 1, "n": 1, "expected": 1},
        {"m": 10, "n": 10, "expected": 48620},
    ]

    for i, test in enumerate(test_cases, 1):
        result = solution.uniquePaths(test["m"], test["n"])
        if result == test["expected"]:
            print(f"Test Case {i} Passed: Output = {result}")
        else:
            print(f"Test Case {i} Failed: Output = {result}, Expected = {test['expected']}")


# Run the test
test_unique_paths()

####################################################
#uniquePathsll.py
# Define the Solution class
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        dp[0][1] = 1

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if obstacleGrid[i - 1][j - 1] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m][n]

def test_unique_paths_with_obstacles():
    solution = Solution()

    test_cases = [
        {
            "input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            "expected": 2
        },
        {
            "input": [[0, 1], [0, 0]],
            "expected": 1
        },
        {
            "input": [[1, 0]],
            "expected": 0
        },
        {
            "input": [[0]],
            "expected": 1
        },
        {
            "input": [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 0]],
            "expected": 0
        }
    ]

    for idx, test in enumerate(test_cases, 1):
        result = solution.uniquePathsWithObstacles(test["input"])
        if result == test["expected"]:
            print(f"Test Case {idx} Passed. Output = {result}")
        else:
            print(f"Test Case {idx} Failed. Output = {result}, Expected = {test['expected']}")

# Run the tests
test_unique_paths_with_obstacles()
