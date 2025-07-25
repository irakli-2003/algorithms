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