# Prove 01
# This is a shell for a classifier program.
# Author: Scott Nicholes
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def main():
    # load the dataset
    iris = loadData()

    # randomize the data
    data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.33)

    # Get the Gaussian Classifier
    #classifier = GaussianNB()
    classifier = HardCodedClassifier()

    # Get the model
    model = classifier.fit(data_train, target_train)

    # Predict Data
    targets_predicted = model.predict(data_test)

    # Compare the targets_predicted to the actual targets
    accuracy = accuracy_score(target_test, targets_predicted)
    print "Accuracy of Prediction: " + str("{:.2f}".format(accuracy * 100)) + "%"

# My Custom Model
class HardCodedModel(object):
    def __init__(self):
        pass

    def predict(self, data_test):
        return [0 for x in range(len(data_test))]

# My Custom Classifier
class HardCodedClassifier(object):
    def __init__(self):
        pass

    def fit(self, data_train, target_train):
        return HardCodedModel()

# Load Data
def loadData():
    data = datasets.load_iris()

    return data

if __name__ == "__main__":
    main()
