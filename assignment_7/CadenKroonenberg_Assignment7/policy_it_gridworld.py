# Author: Caden Kroonenberg
# Date: 11-18-21

import numpy as np

def actionRewardFunction(initialPosition, action):
    # Keep the reward for returning to the termination states at 0 and do not update position
    if initialPosition in terminationStates:
        return initialPosition, 0
    
    reward = rewardSize
    finalPosition = np.array(initialPosition) + np.array(action)
    # Return to initial position on attempts to exit the grid
    if -1 in finalPosition or gridSize in finalPosition: 
        finalPosition = initialPosition
        
    return finalPosition, reward

# Parameters
gamma = 1 # discounting rate
rewardSize = -1
gridSize = 5
terminationStates = [[0,0], [gridSize-1, gridSize-1]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
maxIterations = 5000

# Initialization
valueMap = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
# > [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]

# Print initial values
print("Iteration 0 (Initial Values)")
print(valueMap)
print("")

# Policy Iteration
for it in range(maxIterations):
    copyValueMap = np.copy(valueMap) # copy previous values to make updates for current epoch
    for state in states:
        weightedRewards = 0 # value of cell for next epoch
        for action in actions:
            finalPosition, reward = actionRewardFunction(state, action) # calculate position and reward for action (of possible up, down, left, and right actions)
            weightedRewards += (1/len(actions))*(reward+(gamma*valueMap[finalPosition[0], finalPosition[1]])) # add to value based on reward from action
        copyValueMap[state[0], state[1]] = weightedRewards # set cell value for next epoch
    # Check for convergence
    comparison = valueMap == copyValueMap
    # Print final epoch (after convergence)
    if comparison.all():
        print("Iteration {} (Final Iteration)".format(it+1))
        print(valueMap)
        print("")
        # Stop iterating after convergence
        break
    # update policy array
    valueMap = copyValueMap
    # print policy array for epochs 1 and 10
    if it in [0,9]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")