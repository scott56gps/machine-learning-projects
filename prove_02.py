# Prove 02
# Author: Scott Nicholes
import numpy as np
import pandas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import math
import re
import json

def main():
    print "Welcome to KNN Classifier!"

    # load the dataset
    dataset = loadData()

    # randomize the data
    data_train, data_test, target_train, target_test = train_test_split(dataset["data"], dataset["target"], test_size=0.33)

    # Get the Gaussian Classifier
    #classifier = GaussianNB()
    classifier = HardCodedClassifier()

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(data_train)

    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Get the model
    model = classifier.fit(data_train, target_train, 5)

    # Predict Data
    targets_predicted = model.predict(data_test)

    # Compare the targets_predicted to the actual targets
    accuracy = accuracy_score(target_test, targets_predicted)
    print "Accuracy of Prediction: " + str("{:.2f}".format(accuracy * 100)) + "%"


def readFile():
    fileName = raw_input("Enter filename: ")
    if not re.search("json", fileName):
        print "File must be of type 'json'.  Please try again.  Ending program..."
        quit()
    else:
        readData = open(fileName)
        dataset = json.load(readData)
        return dataset

# My Custom Model
class HardCodedModel(object):
    def __init__(self, data_train, target_train, k):
        self.data_train = data_train
        self.target_train = target_train
        self.k = k
        pass

    # I think here is where I will implement the K-Neighbor algorithm
    def predict(self, data_test):
        predictions = self.kNeighborAlgorithm(data_test, self.data_train, self.target_train, self.k)

        return predictions

    # calculateDistance
    # Returns the Euclidian Distance between the test and input values
    def calculateDistance(self, testingDataPoint, trainingDataPoint):
        return math.sqrt((testingDataPoint[0] - trainingDataPoint[0])**2 + (testingDataPoint[1] - trainingDataPoint[1])**2 + (testingDataPoint[2] - trainingDataPoint[2])**2 + (testingDataPoint[3] - trainingDataPoint[3])**2)

    # kNeighborAlgorithm
    # This function implements the k-nearest neighbors algorithm.
    # self - for Python
    # test_data - The things with which we will test
    # test - The new thing that the algorithm will classify.
    # k - The number of nearest neighbors to return
    def kNeighborAlgorithm(self, data_test, data_train, target_train, k):
        predictions = []

        # Get the Euclidian Distance between the input and each test_data item
        for testingDataPoint in data_test:
            distances = []

            for trainingDataPoint in data_train:
                distances.append(self.calculateDistance(testingDataPoint, trainingDataPoint))

            # Sort the distances from least to greatest
            sortedDistanceIndexes = np.argsort(distances)

            # I have the distances from every test point to every training point.  Now, I will make predictions for every testing point
            # based on the distances I have.

            # Make a prediction
            temp_predictions = []
            for i in range(k):
                # Grab a distance
                temp_predictions.append(distances[sortedDistanceIndexes[i]])

            # Take the average of the predictions
            # Round the integer and put it on the predictions
            predictions.append(int(round(sum(temp_predictions) / len(temp_predictions))))

        # Convert back to a numpy array
        predictions = np.array(predictions)
        return predictions

# My Custom Classifier
class HardCodedClassifier(object):
    def __init__(self):
        pass

    def fit(self, data_train, target_train, k):
        return HardCodedModel(data_train, target_train, k)

# Load Data
def loadData():
    choice = input("Select a choice: 1. Use IRIS data  2. Digits data: ")
    if choice == 1:
        data = datasets.load_iris()
    elif choice == 2:
        # For another day...
        #data = readFile()
        data = datasets.load_digits(n_class=2)
    else:
        print "Incorrect choice.  Please use number 1 or 2.  Ending program..."
        quit()
    return data

if __name__ == "__main__":
    main()
