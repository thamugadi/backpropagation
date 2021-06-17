import numpy as np

def reLU(x):
    if x > 0:
        return x
    return 0
def dreLU(x):
    if x > 0:
        return 1
    return 0
def datan(x):
    return 1/(1+(x**2))
def MSE(A,B):
    return ((A - B)**2).mean(axis=0)
def MSE_vec(A,B):
    return ((A - B)**2)
def dMSE(A,B):
    return (2*(A - B)).mean(axis=0)
def dMSE_vec(A,B):
    return (2*(A - B))
def identity(x):
    return x

def evaluateNetwork(W, B, firstLayer, activation):
    if len(W) == 0:
        return []
    secondLayer = np.vectorize(activation)(np.dot(W[0], firstLayer)+B[0])
    return [secondLayer]+evaluateNetwork(W[1:],B[1:], secondLayer, activation)

def evaluateAggregated(W,B, firstLayer, activation):
    if len(W) == 0:
        return []
    secondLayer = np.vectorize(activation)(np.dot(W[0], firstLayer)+B[0])
    secondLayerNonAct = (np.dot(W[0], firstLayer)+B[0])
    return [secondLayerNonAct]+evaluateAggregated(W[1:],B[1:], secondLayer, activation)

def backpropagate(W,B, firstLayer, activation, derivative, dCost, correct):
    network = evaluateNetwork(W, B, firstLayer, activation)
    agnetwork = evaluateAggregated(W,B, firstLayer, activation)
    n = len(network)-1
    toBeReversed = []
    error = np.vectorize(dCost)(network[n], correct) * np.vectorize(derivative)(agnetwork[n])
    toBeReversed.append(error)
    if n == 0:
        return toBeReversed
    for i in range(n-1, 0, -1):
        error = np.dot(np.transpose(W[i+1]), error)*np.vectorize(derivative)(agnetwork[i])
        toBeReversed.append(error)
    toBeReversed.append(np.dot(np.transpose(W[0]), error)*np.vectorize(derivative)(firstLayer))
    toBeReversed.reverse()
    return (toBeReversed, network)

def updateWeights(W,B, firstLayer, activation, derivative, dCost, correct, rate):
    (errors, network) = backpropagate(W,B,firstLayer,activation,derivative,dCost,correct)

    NW = []
    n = len(W)
    NW.append(W[0] - rate*np.array(firstLayer)*errors[0])
    for i in range(1, n):
        NW.append(W[i] - rate*network[i-1]*errors[i])
    return NW

def updateBiases(W,B, firstLayer, activation, derivative, dCost, correct, rate):
    (errors, network) = backpropagate(W,B,firstLayer,activation,derivative,dCost,correct)
    NB = []
    n = len(B)
    for i in range(n):
        NB.append(B[i] - rate*errors[i])
    return NB
