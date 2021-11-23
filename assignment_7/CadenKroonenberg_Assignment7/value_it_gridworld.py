# Author: Caden Kroonenberg
# Date: 11-22-21

import numpy as np

def actionRewardFunction(initialPosition, action):
    
    if initialPosition in terminationStates:
        return initialPosition, 0
    
    # reward = rewardSize
    finalPosition = np.array(initialPosition) + np.array(action)
    if -1 in finalPosition or gridSize in finalPosition: 
        finalPosition = initialPosition
        
    return finalPosition, -1

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

# Print epoch 0 values (initial values)
print("Iteration 0 (Initial Values)")
print(valueMap)
print("")

# Policy Iteration
for it in range(maxIterations):
    copyValueMap = np.copy(valueMap)
    for state in states:
        weightedRewards = np.empty(shape=0)
        for action in actions:
            finalPosition, reward = actionRewardFunction(state, action)
            weightedReward = reward + valueMap[finalPosition[0], finalPosition[1]]
            weightedRewards = np.insert(weightedRewards, weightedRewards.size, weightedReward)
        
        copyValueMap[state[0], state[1]] = np.max(weightedRewards)
    # Check for convergence
    comparison = valueMap == copyValueMap
    # Print values if convergence was reached
    if comparison.all():
        print("Iteration {} (Final Iteration)".format(it+1))
        print(valueMap)
        print("")
        # Stop iterating after convergence
        break
    # Update policy array
    valueMap = copyValueMap
    # Print policy array for epochs 1 and 2
    if it in [0,1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")