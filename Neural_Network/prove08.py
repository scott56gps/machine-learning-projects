import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.offline as py
import plotly.graph_objs as go

class NeuralNetworkClassifier(object):
    def __init__(self, numNodes, numHiddenLayers, learningRate):
        self.numNodes = numNodes
        self.numHiddenLayers = numHiddenLayers
        self.learningRate = learningRate

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
    # numTargets is the number of possible targets
    def fit(self, numInputs, numTargets):
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
        numTargetNodes = numTargets

        # If there are only 2 possibilities, we simply use 1 target node to indicate which one we predict
        if numTargetNodes == 2:
            numTargetNodes = 1
        layers.append(np.array(self.makeLayer(self.numNodes + 1, numTargetNodes)))
        return layers

    # makeLayer
    def makeLayer(self, numInputs, numNodes):
        # We will create a randomWeight for each of the inputs associated with each node
        randomWeight = lambda a : np.random.uniform(-1,1)
        layer = [[randomWeight(j) for j in range(numNodes)] for i in range(numInputs)]
        return layer

    # Make Error Vector For Hidden Layer
    def makeErrorVectorForHiddenLayer(self, weightLayer, errorVector, activationVector):
        # Account for the bias node
        activationVector = np.append(activationVector, -1)

        # 1. Calculate the outer product of weightLayer and errorVector (Example: 5x1 . 1x1 -> 5x1, OR (from hidden layer to hidden layer) 5x4 . 4x1)
        weightAndError = np.dot(weightLayer, errorVector)
        weightAndError = weightAndError.reshape(len(weightLayer), 1)

        # 2. SUM each row of 1. (I don't know if I need this...  I will not do it for right now.)
        weightAndError = np.matrix(weightAndError).sum(axis=1)

        # 3. activationVector * (1 - each value of activationVector)  (Example: 5x1 *(NOT DOT PRODUCT) (5x1 - 5x1))
        activationCalculation = np.multiply(activationVector, np.subtract([1 for i in range(len(activationVector))], activationVector))

        # 4. Calculate dot product of 2. and 3.
        returnErrorVector = np.outer(activationCalculation, weightAndError)
        returnErrorVector = np.delete(returnErrorVector, len(returnErrorVector) - 1, 0)
        returnErrorVector = np.matrix(returnErrorVector).sum(axis=1)
        return returnErrorVector

    # Make Hidden Layers Error Vectors
    def makeHiddenLayersErrorVectors(self, layers, activationVectors, outputLayerErrorVector):
        # Make the Error Vector for the last layer first, since we need that information to process the rest of the layers
        hiddenLayerErrorVectors = []
        hiddenLayerErrorVectors.append(self.makeErrorVectorForHiddenLayer(layers[-1], outputLayerErrorVector, activationVectors[-2]))

        for i in reversed(xrange(len(layers) - 1)):
            # Make the error vector for each layer
            errorVector = self.makeErrorVectorForHiddenLayer(layers[i], hiddenLayerErrorVectors[-1], activationVectors[i])

            # Concatenate the shaped vector onto the hiddenLayerErrorVectors
            hiddenLayerErrorVectors.append(errorVector)

        # Remove the last element of the hiddenLayers, as it calculated the error for the inputs
        return hiddenLayerErrorVectors[:-1]

    def computeOutputError(self, activationVector, targetVector):
        def outputErrorFunction(activationValue, targetValue):
            return activationValue * (1 - activationValue) * (activationValue - targetValue)

        vectorFunction = np.vectorize(outputErrorFunction)
        return vectorFunction(activationVector, targetVector)

    # Make Error Matrix
    def getErrorMatrix(self, layers, activationVectors, targetVector):
        # Make output error vector
        outputErrorVector = self.computeOutputError(activationVectors[-1], targetVector)

        # Make Hidden Layers vectors
        hiddenLayerErrorVectors = self.makeHiddenLayersErrorVectors(layers, activationVectors, outputErrorVector)

        # Put Hidden Lyaers Vectors and output layer error vector together into a matrix
        hiddenLayerErrorVectors.append(outputErrorVector)
        return hiddenLayerErrorVectors

    def updateWeights(self, layer, errorVector, activationVector):
        # Put Bias input in
        activationVector = np.append(activationVector, -1)

        # Right Term means the right term of w_ij - N*d_j*a_i
        rightTerm = np.outer(errorVector, activationVector)

        # Multiply Right Term by the Learning Rate
        rightTerm = np.multiply(self.learningRate, rightTerm)

        # Reshape the right term so that it can be subtracted
        rightTerm = rightTerm.reshape(len(rightTerm[0]), len(rightTerm))

        # Subtract at once the layer and the rightTerm
        updatedWeights = np.subtract(layer, rightTerm)

        return updatedWeights

    # Propogate Backward
    # This function will execute the entire back propogation process.
    # It will output the updated weights of all the layers of the Neural Network.
    def propogateBackward(self, layers, activationVectors, targetVector):
        # Get the Error Matrix
        errorVectors = self.getErrorMatrix(layers, activationVectors, targetVector)

        for i in range(len(layers)):
            layers[i] = self.updateWeights(layers[i], errorVectors[i], activationVectors[i])

        return layers

    def computeLayer(self, inputVector, layer):
        sigmoid = lambda a : 1.0 / (1 + math.exp(-(a)))
        hValues = np.dot(inputVector, layer)
        return [sigmoid(hValue) for hValue in hValues]

    def propogateForward(self, inputVector, layer):
        # Account for Bias Input
        inputVector = np.append(inputVector, -1)
        return self.computeLayer(inputVector, layer)

    def makePrediction(self, finalActivationVector):
        for i in range(len(finalActivationVector)):
            finalActivationVector[i] = round(finalActivationVector[i])

        return finalActivationVector

    def train(self, layers, inputVectors, targetVector):
        for j in range(len(inputVectors)):
            # We start by appending the inputs as the first "activation" vector
            activationVectors = [inputVectors[j]]
            for i in range(len(layers)):
                if i == 0:
                    activationVectors.append(self.propogateForward(inputVectors[j], layers[i]))
                else:
                    activationVectors.append(self.propogateForward(activationVectors[i], layers[i]))

            # OK, now that we have propogated forward, it is time to propogate backward
            layers = self.propogateBackward(layers, activationVectors, targetVector[j])

        return layers

    def predict(self, layers, test_input_vectors, test_target_vectors):
        # Propogate Forward and make a prediction
        predictions = []
        accuracies = []
        for i in range(len(test_input_vectors)):
            activationVectors = [test_input_vectors[i]]
            for j in range(len(layers)):
                if j == 0:
                    activationVectors.append(self.propogateForward(test_input_vectors[i], layers[j]))
                else:
                    activationVectors.append(self.propogateForward(activationVectors[j], layers[j]))

            # Now, we make a prediction
            predictions.append(self.makePrediction(activationVectors[-1]))

            accuracies.append(getAccuracy(test_target_vectors[i][0], predictions[i][0]))
        return predictions, accuracies

def readData():
    filename = raw_input("Enter filename: ")
    data = np.loadtxt(filename, delimiter=",")
    print data

def loadData():
    data = datasets.load_iris()

    return data

def discretizeTargets(target_values, numChoices):
    discretizedTargets = []
    for target in target_values:
        discretizedTarget = np.zeros(numChoices)

        # Insert a 1 in the place that the target specifies
        discretizedTarget[target] = 1

        discretizedTargets.append(discretizedTarget)
    return discretizedTargets

def getAccuracy(target_test, predictions):
    if isinstance(target_test, int) or isinstance(target_test, float):
        return accuracy_score([target_test], [predictions])
    else:
        accuracies = np.array([])
        for i in range(len(target_test)):
            accuracy = accuracy_score(target_test[i], predictions[i])
            accuracies = np.append(accuracies, accuracy)
        return np.mean(accuracies)

def plotAccuracies(accuracies):
    trace = go.Scatter(x = [i+1 for i in range(len(accuracies))], y = accuracies)
    data = [trace]
    py.plot(data, filename='neural-network-scatter2.html')

# load the dataset
iris = loadData()

# randomize the data
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.33)
train_input_vectors = data_train
test_input_vectors = data_test
numTargets = 3

# I created this dataset.  A fake caterpillar dataset.
# targets: [1: Zebra Caterpillar, 2: Green Caterpillar, 3: The Gargantua]
#train_input_vectors = np.array([[20, 5], [15, 2], [19, 4], [10, 4], [25, 5], [22, 6]])
#target_train = np.array([[1], [2], [1], [2], [1], [1]])
#test_input_vectors = np.array([[15, 3], [10, 5], [20, 9]])
#target_test = np.array([[2], [2], [1]])

target_train = discretizeTargets(target_train, numTargets)
target_test = discretizeTargets(target_test, numTargets)

neuralNetClassifier = NeuralNetworkClassifier(4, 4, .1)
neuralNetwork = neuralNetClassifier.fit(len(train_input_vectors[0]), numTargets)
neuralNetwork = neuralNetClassifier.train(neuralNetwork, train_input_vectors, target_train)
predictions, accuracies = neuralNetClassifier.predict(neuralNetwork, test_input_vectors, target_test)
accuracy = getAccuracy(target_test, predictions)
plotAccuracies(accuracies)
print "Accuracy of Prediction: " + str("{:.2f}".format(accuracy * 100)) + "%"
