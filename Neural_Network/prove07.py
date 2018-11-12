import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NeuralNetworkClassifier(object):
    def __init__(self, numNodes, numHiddenLayers):
        self.numNodes = numNodes
        self.numHiddenLayers = numHiddenLayers

    # fit
    # Target Vector is the vector of all the targets for all the input vectors
    # Example:
    # x1 | x2 | target
    # 1    2    1
    # 4    8    2
    # 2    1    2
    # 1    9    1
    #
    # targetVector is the entire column of target
    def fit(self, numInputs, targetVector):
        # Construct the network of weights
        layers = []
        for i in range(self.numHiddenLayers):
            layer = []
            if i == 0:
                layer = self.makeLayer(numInputs + 1, self.numNodes)
            else:
                layer = self.makeLayer(self.numNodes + 1, self.numNodes)

            layers.append(np.array(layer))

        # Make the output layer by making a layer with the number of nodes corresponding to unique values in the target vector
        numTargetNodes = len(set(targetVector))

        # If there are only 2 possibilities, we simply use 1 target node to indicate which one we predict
        if numTargetNodes == 2:
            numTargetNodes = 1
        layers.append(np.array(self.makeLayer(self.numNodes + 1, numTargetNodes)))
        return layers

    def propogateForward(self, inputVector, layer):
        # Account for Bias Input
        inputVector = np.append(inputVector, -1)
        return self.computeLayer(inputVector, layer)

    def makePrediction(self, finalActivationVector):
        for i in range(len(finalActivationVector)):
            finalActivationVector[i] = round(finalActivationVector[i])

        return finalActivationVector

    # calculateOutputError
    # This function calculates the error of the output layer node(s).
    #
    # activationValue : float - The activation value for a given output layer node
    # targetValue : float - The target value for a given output layer node
    #
    # return : The error (delta j) for this output layer node
    def calculateOutputError(activationValue, targetValue):
        return activationValue * ((1 - activationValue) * (activationValue - targetValue))

    def predict(self, layers, inputVectors, targetVector):
        predictions = []
        for inputVector in inputVectors:
            activationVectors = []
            for i in range(len(layers)):
                if i == 0:
                    activationVectors.append(self.propogateForward(inputVector, layers[i]))
                else:
                    activationVectors.append(self.propogateForward(activationVectors[i-1], layers[i]))

            predictions.append(self.makePrediction(activationVectors[-1]))

        return np.array(predictions)

    # makeLayer
    def makeLayer(self, numInputs, numNodes):
        # We will create a randomWeight for each of the inputs associated with each node
        randomWeight = lambda a : np.random.uniform(-1,1)
        layer = [[randomWeight(j) for j in range(numNodes)] for i in range(numInputs)]
        return layer

    def computeLayer(self, inputVector, layer):
        sigmoid = lambda a : 1.0 / (1 + math.exp(-(a)))
        hValues = np.dot(inputVector, layer)
        return [sigmoid(hValue) for hValue in hValues]

def readData():
    filename = raw_input("Enter filename: ")
    data = np.loadtxt(filename, delimiter=",")
    print data

def loadData():
    data = datasets.load_iris()

    return data

def getAccuracy(target_train, predictions):
    return accuracy_score(target_train, predictions)

# load the dataset
iris = loadData()

# randomize the data
#data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.33)
#inputVectors = data_train

# I created this dataset.  A fake caterpillar dataset.
# targets: [1: Zebra Caterpillar, 2: Green Caterpillar, 3: The Gargantua]
inputVectors = np.array([[20, 5], [15, 2], [19, 4], [10, 4], [25, 5], [22, 6]])
target_train = np.array([1, 2, 1, 2, 1, 1])

# I created this dataset.
#inputVectors = np.array([[1, 2], [3, 4]])
#target_train = [1, 2]
neuralNet = NeuralNetworkClassifier(4, 2)
layers = neuralNet.fit(len(inputVectors[0]), target_train)
predictions = neuralNet.predict(layers, inputVectors, target_train)
accuracy = getAccuracy(target_train, predictions)
print "Accuracy of Prediction: " + str("{:.2f}".format(accuracy * 100)) + "%"
