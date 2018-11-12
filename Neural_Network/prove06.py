import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class NeuralNetwork(object):
    def __init__(self, inputVector, targetValues):
        # Account for Bias Input
        self.inputLayer =  self.makeLayer(len(inputVector) + 1, len(targetValues))
        self.targetValues = targetValues

    # makeLayer
    def makeLayer(self, numInputs, numNodes):
        # We will create a randomWeight for each of the inputs associated with each node
        randomWeight = lambda a : np.random.uniform(-1,1)
        layer = [[randomWeight(j) for j in range(numNodes)] for i in range(numInputs)]
        return np.array(layer)

    def computeLayer(self, inputVector, layer):
        inputVector = np.append(inputVector,-1)
        return np.dot(inputVector, layer)

def readData():
    filename = raw_input("Enter filename: ")
    data = np.loadtxt(filename, delimiter=",")
    print data

def loadData():
    data = datasets.load_iris()

    return data

# load the dataset
iris = loadData()

# randomize the data
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.33)


#data = readData()
inputVector = data_train
#print len(inputVector[0])
#print target_train
#inputVector = np.array([20.4, 72.1])
neuralNet = NeuralNetwork(inputVector[0], [target_train[0]])
print neuralNet.computeLayer(inputVector[0], neuralNet.inputLayer)
