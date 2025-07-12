def toplogicalSortUtil(v, adj, visited, stack):
    visited[v] = True

    for i in adj[v]:
    # შეავსეთ გამოტოვებული ფრაგმენტი
        if not visited[i]:
            toplogicalSortUtil(i, adj, visited, stack)
    stack.append(v)


def constructadj(V, edges):
    adj = [[] for _ in range(V)]
    for it in edges:
        adj[it[0]].append(it[1])
    return adj


def toplogicalSort(V, edges):
    stack = []
    visited = [False] * V

    adj = constructadj(V, edges)
    for i in range(V):
        if not visited[i]:
            toplogicalSortUtil(i, adj, visited, stack)
    return stack[::-1]

v=7
edges=[[0,1],[0,2],[2,3],[3,4],[3,5],[0,6]]

ans = toplogicalSort(v, edges)

print(" ".join(map(str, ans)))