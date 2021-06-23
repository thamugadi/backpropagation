import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(x):
    return (sigmoid(x))*(1-sigmoid(x))
def dMSE(A,B):
    return 2*(A-B)

class VanillaLayer:
    def __init__(self, inputLength,outputLength, activation,derivative):
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.afunc = activation
        self.dfunc = derivative
        self.wmatrix = np.random.rand(outputLength,inputLength)
        self.bvector = np.random.rand(outputLength)
    def evaluate(self, input):
        self.activation = self.afunc(np.dot(input, self.wmatrix)+self.bvector)
        self.h = np.dot(input, self.wmatrix)+self.bvector


class HiddenLayer(VanillaLayer):
    def computeError(self, nextlayer):
        self.error = np.dot(np.transpose(nextlayer.wmatrix),nextlayer.error)*self.dfunc(self.h)


class OutputLayer(VanillaLayer):
    def computeError(self, dCost, input, correct):
        self.error = dCost(self.activation,correct)*self.dfunc(self.h)


class VanillaNetwork:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
    def evaluate(self, input):
        self.network = []
        self.hnetwork = []
        output = np.array(np.array(input))
        self.network.append(np.array(input))
        self.hnetwork.append(np.array(input))
        for layer in self.layers:
            layer.evaluate(output)
            output = layer.activation
            h = layer.h
            self.network.append(output)
            self.hnetwork.append(h)
        self.houtput = h
        self.output = output
        return output
    def backpropagate(self, input, correct, dCost):
        self.evaluate(np.array(input))
        self.gradients = []
        n = len(self.layers)
        error = self.layers[n-1].computeError(dCost, np.array(input), np.array(correct))
        for layer, nextlayer in zip(self.layers[::-1][1:], self.layers[::-1]):
            layer.computeError(nextlayer)
        for layer in self.layers:
            self.gradients.append(layer.error)
    def adjustWeights(self, rate):
        for (layer, a) in zip(self.layers, self.network):
            layer.wmatrix = layer.wmatrix - rate*a*layer.error
    def adjustBiases(self,rate):
        for layer in self.layers:
            layer.bvector = layer.bvector - rate*layer.error

H1 = HiddenLayer(3,3,sigmoid,dsigmoid)
O1 = OutputLayer(3,3,sigmoid,dsigmoid)

N = VanillaNetwork()
N.add(H1)
N.add(O1)
for _ in range(1001):
    N.backpropagate([0,0,1],[1,0,0],dMSE)
    N.adjustWeights(0.2)
