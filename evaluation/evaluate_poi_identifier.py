#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import cross_validation
from sklearn import tree
from sklearn import metrics

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels,test_size = 0.3, random_state = 42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

print y_test
print clf.predict(X_test)
print len(X_test)
print(clf.score(X_test, y_test))

labels_predicted = clf.predict(X_test)
labels_predicted = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# print metrics.confusion_matrix(y_test, labels_predicted)
print metrics.precision_score(y_test, labels_predicted, average = "binary")
# print metrics.recall_score(y_test, labels_predicted, average = "binary")

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

# print metrics.confusion_matrix(true_labels, predictions)
# print metrics.precision_score(true_labels, predictions, average='binary')
# print metrics.recall_score(true_labels, predictions, average='binary')
