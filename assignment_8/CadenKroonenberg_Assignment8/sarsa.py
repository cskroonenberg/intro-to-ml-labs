# Author: Caden Kroonenberg
# Date: 12-8-21

import numpy as np
import random

# SARSA Algorithm

# Check if a position is a termination State
def check(position):
    return any(np.array_equal(x, position) for x in terminationStates)

# Convert coordinate position to a state #
def stateNum(position):
    return int(position[0]*gridSize + position[1])

# Retrieve coordiante position from a state #
def findPosition(state):
    return [state//gridSize, state%gridSize]

# Print Q and R matrices
def print_arr(arr):
    print("S\A", end="\t")
    for i in range(len(arr)):
        print(i, end="\t")
    print("")
    for i in range(len(arr)):
        print(i, end="\t")
        for j in range(len(arr[0])):
            print(arr[i][j], end="\t")
        print("")

# Check for convergence after each episode
def checkConvergence(copyQ, Q):
    # Check that each state has Q values for each possible action
    for i in range(1,numStates-1):
        for j in range(0,numStates):
            if R[i][j] != -1 and Q[i][j] != 0: # if no Q values are recorded for a given state (i.e. Q[i][all states] = 0) ...
                break 
            if j == numStates-1: # ... Return false, each (non terminal) state should be visited at least once
                return False
    # All actions have been evaluated. Check if Q values are still updating from iteration to iteration - return true if not
    comparison = Q == copyQ
    return comparison.all()

# Parameters
gamma = 0.9 # discounting rate
rewardSize = -1 # Reward size
gridSize = 5
terminationStates = np.array([[0,0], [gridSize-1, gridSize-1]])
actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
maxEpisodes = 100000

numStates = pow(gridSize, 2) # Number of unique states in gridSize x gridSize grid world
Q = np.zeros((numStates, numStates))
R = np.full((numStates, numStates), -1)

# Initialize R
for state in range(25):
    if check(findPosition(state)):
        R[int(state)][int(state)] = 100
    for action in actions:
        neighbor = np.array(findPosition(state)) + np.array(action)
        if not(-1 in neighbor or gridSize in neighbor): 
            R[int(state)][stateNum(neighbor)] = 100 if check(neighbor) else 0

# Print Rewards Matrix
print("SARSA Rewards Matrix (R)")
print_arr(R)

# Print Initial Value Matrix
print("Episode 0 SARSA Value Matrix (V) (Initial Values)")
print_arr(Q)

# Episode Iteration
for episode in range(1,maxEpisodes):
    # Copy Q values to check for convergence at end of episode
    copyQ = np.copy(Q)

    # Randomly select initial state
    state = random.randint(0,numStates-1)

    while not check(findPosition(state)): # While not at a terminal state ...
        # Find all possible next states from current state
        actions = np.array([])
        for i in range(numStates):
            if R[state][i] != -1:
                actions = np.append(actions, i)
        
        # Find the action with the maximum estimated value
        next_actions = np.array([]) # Next action consideration pool
        next_actions = np.append(next_actions, int(actions[0]))
        for action in np.delete(actions, 0):
            if Q[state][int(action)] > Q[state][int(next_actions[0])]: # If estimated value of action > estimated value of next move ...
                next_actions = np.array([action]) # Only consider this action
            elif Q[state][int(action)] == Q[state][int(next_actions[0])]: # If Q value is equal for each action ...
                next_actions = np.append(next_actions, action) # Add action to consideration pool
        action = int(random.choice(next_actions))

        # Find all possible next states from next state
        next_actions = np.array([])
        for i in range(numStates):
            if R[action][i] != -1:
                next_actions = np.append(next_actions, i)

        # find max Q for next states possible actions
        max = 0
        for next in next_actions:
            if Q[action][int(next)] > max:
                max = Q[action][int(next)]

        # Update Q(s,a) using Bellman Equation
        Q[state][action] = R[state][action] + gamma*max
        # Update state
        state = action

    # Round values for simplicity
    Q = np.around(Q, decimals=3)
    # Check for convergence
    if checkConvergence(copyQ, Q):
        print("\nEpisode {} SARSA Value Matrix (V) (Final Episode - CONVERGENCE DETECTED)".format(episode))
        print_arr(Q)
        exit()

    # Print on episode 1 and 10
    if episode in [1,10]:
        print("\nEpisode {} SARSA Value Matrix (V)".format(episode))
        print_arr(Q)

    # If max episode # has been reached without convergence, print out final values
    if episode == maxEpisodes-1:
        print("\nEpisode {} SARSA Value Matrix (V) (Final Episode - NO CONVERGENCE DETECTED)".format(episode))
        print_arr(Q)
        exit()