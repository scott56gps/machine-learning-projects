import math
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# A Tree must have nodes
class TreeNode(object):
    def __init__(self):
        self.data = ""
        self.children = {}

# I presume that the 'Model' will be the decision tree itself
class DecisionTreeModel(object):
    def __init__(self, tree={}):
        self.tree = tree

    def traverseTree(self, requests, tree, attribute):
        valueForCurrentAttribute = requests[attribute]
        if valueForCurrentAttribute not in tree[attribute].keys():
            # Find another value to send along
            # For now, just send nothing
            return "NO_VALUE"
        elif (type(tree[attribute][valueForCurrentAttribute]) is str):
            return tree[attribute][valueForCurrentAttribute]
        else:
            return self.traverseTree(requests, tree[attribute][valueForCurrentAttribute], tree[attribute][valueForCurrentAttribute].keys()[0])

    def predict(self, targets):
        return self.traverseTree(targets, self.tree, self.tree.keys()[0])

# My Decision Tree Classifier
class DecisionTreeClassifier(object):
    def __init__(self):
        pass

    def calculateEntropy(self, labels):
        # Make Lambda function
        calculate_probability = lambda unique_label : -float(labels.count(unique_label)) / len(labels) * math.log(float(labels.count(unique_label)) / len(labels), 2)

        # Make the set of unique labels
        unique_labels = list(set(labels))

        # Calculate the total entropy based upon each label
        entropy = sum([calculate_probability(unique_label) for unique_label in unique_labels])

        return entropy

    # This function takes an attribute (such as income) and calculates its Information Gain
    def calculateInfoGain(self, feature, targets):
        # OK, so get a unique list of the attributes
        unique_attributes = list(set(feature))
        entropies = {}

        # For each unique attribute in the attributes
        for unique_attribute in unique_attributes:
            labels = []
            attribute_count = 0
            # Gather the targets associated with each of the attributes
            for i in range(len(feature)):
                if (feature[i] == unique_attribute):
                    # Append the label for this feature onto the labels list
                    labels.append(targets[i])
                    attribute_count += 1

            # IF all the labels are the same, then we create a leaf node for this attribute

            # Get the entropy for this attribute's labels
            entropy = self.calculateEntropy(labels)

            # Calculate Weighted Value for this attribute's entropy
            entropies[unique_attribute] = (float(attribute_count) / len(feature)) * entropy

        return sum(entropies.values())

    def buildTree(self, features, targets):
        if len(list(set(targets))) == 1:
            return targets[0]
        elif len(features) == 0:
            return targets[0]
        # Find the highest info gain (lowest entropy)
        entropies = {}
        for feature in features.keys():
            # Calculate the Information Gain
            entropies[feature] = self.calculateInfoGain(features[feature], targets)

        # Build the tree with the attribute that has the highest info gain (lowest entropy)
        bestFeature = min(entropies, key=entropies.get)
        tree = {bestFeature: []}

        # Use the unique values for the best feature to be branches of this node in the tree
        tree[bestFeature] = dict.fromkeys(list(set(features[bestFeature])), {})

        for branch in tree[bestFeature]:
            newFeatures = {}
            newTargets = []
            for feature in features.keys():
                newFeatures[feature] = []

            for i in range(len(targets)):
                if (features[bestFeature][i] == branch):
                    newTargets.append(targets[i])
                    for feature in features.keys():
                        newFeatures[feature].append(features[feature][i])

            del newFeatures[bestFeature]
            subtree = self.buildTree(newFeatures, newTargets)
            tree[bestFeature][branch] = subtree

        return tree

    def fit(self, data_train, data_target):
        tree = self.buildTree(data_train, data_target)
        return DecisionTreeModel(tree)

# This function reads a file in from user input
def readFile():
    dataFileName = raw_input("Enter filename for the data: ")
    dataset = pd.read_csv(dataFileName, header=0, skipinitialspace=True)
    return dataset

def main():
    # Load the data
    data_train_frame = readFile()
    targets = data_train_frame.iloc[:, -1].values
    data_train_frame = data_train_frame.drop(data_train_frame.columns[-1], axis=1)

    training_data, testing_data, training_targets, testing_targets = train_test_split(data_train_frame, targets, test_size=0.33)

    # Format the data into a dictionary
    data_train = {}
    for column in training_data.columns:
        data_train[column] = list(training_data.loc[:, column].values)

    #data_train = {'credit_score': ['Good', 'Good', 'Good', 'Good', 'Average', 'Average', 'Average', 'Average', 'Low', 'Low', 'Low', 'Low'],
    #              'income': ['High', 'High', 'Low', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'High', 'Low', 'Low'],
    #              'collateral': ['Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Poor', 'Good', 'Good', 'Poor', 'Good', 'Poor']}
    #targets = ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No']
    #value_names = ['credit_score', 'income', 'collateral']

    data_test_dict = testing_data.to_dict('records')

    classifier = DecisionTreeClassifier()
    model = classifier.fit(data_train, training_targets)

    # OK, so now each of the data_train_values should correspond with a target
    targets_predicted = []
    for testing_value in data_test_dict:
        targets_predicted.append(model.predict(testing_value))

    accuracy = accuracy_score(testing_targets, targets_predicted)
    print "Accuracy: " + str("{:.2f}".format(accuracy * 100)) + "%"

if __name__ == "__main__":
    main()
