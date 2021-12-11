# Author: Caden Kroonenberg
# Date: 12-7-21

import numpy as np
import random
# Monte Carlo First Visit Algorithm

# Check if a position is a termination State
def check(position):
    return any(np.array_equal(x, position) for x in terminationStates)

# Apply action
def actionRewardFunction(initialPosition, action):
    reward = rewardSize
    finalPosition = np.array(initialPosition) + np.array(action)
    # Return to initial position on attempts to exit the grid
    if -1 in finalPosition or gridSize in finalPosition: 
        finalPosition = initialPosition
    # Reward for returning to the termination states is 0
    if check(finalPosition):
        reward = 0
    return finalPosition, reward

# Convert coordinate position to a state #
def stateNum(position):
    return position[0]*gridSize + position[1]

# Retrieve coordiante position from a state #
def findPosition(state):
    return [state//gridSize, state%gridSize]

# print N(s), S(s), and V(s)
def printVals():
    print("N(s)")
    print(N_s)
    print("S(s)")
    print(S_s)
    print("V(s)")
    print(V_s)

# Parameters
gamma = 0.9 # discounting rate
rewardSize = -1 # Reward size
gridSize = 5
terminationStates = np.array([[0,0], [gridSize-1, gridSize-1]])
actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
maxEpochs = 100000

states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
# > [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]

N_s = np.zeros((gridSize, gridSize))
S_s = np.zeros((gridSize, gridSize))
V_s = np.zeros((gridSize, gridSize))

print("Episode 0 (Initial Values - First Visit Method)")
printVals()

# Episode Iteration
for epoch in range(1,maxEpochs):
    # Copy V(s) matrix to check for convergence
    copyV_s = np.copy(V_s)

    # Randomly select initial state that isn't a termination state
    position = states[random.randint(1,len(states)-2)]

    # Table to keep track of k, s, r, γ, and G(s) values
    G_s_table = np.zeros((0, 5))
    
    # Array to keep track of which states have already been evaluated (specifically for first-visit MC)
    visited = np.zeros((gridSize, gridSize))
    k = 1

    # Add initial row to G_s table
    row = [k,stateNum(position), 0 if check(position) else rewardSize, gamma, 0]
    G_s_table = np.vstack([G_s_table,row])

    while not check(position): # While not at a terminal state ...
        k += 1
        action = random.choice(actions) # Randomly select action (up, down, left, right)
        position, reward = actionRewardFunction(position, action)
        # Record k, s, r, γ values for action
        row = [k,stateNum(position), reward, gamma, 0]
        G_s_table = np.vstack([G_s_table,row])

    num_actions = G_s_table.shape[0]
    # Calculate G(s) for each row and N(s), S(s) for each position
    for i in range(num_actions-1):
        k = int(G_s_table[i][0])
        s = int(G_s_table[i][1])
        state = findPosition(s)
        # For each row in the k, s, r, γ, G(s) table ...
        for j in range(int(num_actions-k) + 1):
            # G(s) = r(t+1)+ γ*r(t+2) + γ^(2)*r(t+3) + … + γ^(k-1)*r(k)
            current = G_s_table[int(k + j - 1)]
            G_s_table[i][4] += pow(current[3], j)*(current[2])
        # Update N(s) unless s is a terminal state
        if not check(state) and not visited[state[0]][state[1]]:
            N_s[state[0]][state[1]] += 1
            # S(s) = sum of first-visited G(s) for each s
            S_s[state[0]][state[1]] += G_s_table[i][4]
        visited[state[0]][state[1]] = 1
        
    # Calculate V(s) = S(s)/N(s) for each state
    for s in range(gridSize*gridSize):
        state = findPosition(s)
        if int(N_s[state[0]][state[1]]) > 0:
            V_s[state[0]][state[1]] = S_s[state[0]][state[1]]/int(N_s[state[0]][state[1]])
    
    # Round to 2rd decimal place to ease convergence detecton
    V_s = np.around(V_s, decimals=2)
    S_s = np.around(S_s, decimals=2)
    # Check for convergence
    comparison = V_s == copyV_s
    if comparison.all():
        print("\nEpisode {} (Final Episode - First Visit Method - CONVERGENCE DETECTED)".format(epoch))
        printVals()
        print("k, s, r, γ, G(s) values:")
        print(G_s_table)
        exit()
    # Print on epoch 1 and 10
    if epoch in [1,10]:
        print("\nEpisode {} (First Visit Method)".format(epoch))
        printVals()
        print("k, s, r, γ, G(s) values:")
        print(G_s_table)

    # If max epoch # has been reached without convergence, print out final values
    if epoch == maxEpochs-1:
        print("\nEpisode {} (Final Episode - Every Visit Method - NO CONVERGENCE DETECTED)".format(epoch))
        printVals()
        exit()