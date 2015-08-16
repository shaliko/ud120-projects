#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import enron_tools
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from time import time

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi', 'salary', 'shared_receipt_with_poi'] # 1
# features_list = ['poi','from_poi_to_this_person'] # 2
# features_list = [
#     'poi',
#     'salary',
#     'bonus',
#     'fraction_from_poi_email',
#     'fraction_to_poi_email',
#     'deferral_payments',
#     'total_payments',
#     'loan_advances',
#     'restricted_stock_deferred',
#     # 'deferred_income',
#     # 'total_stock_value',
#     # 'expenses',
#     # 'exercised_stock_options',
#     # 'long_term_incentive',
#     # 'shared_receipt_with_poi',
#     # 'restricted_stock',
#     # 'director_fees'
# ]
# features_list = ['poi', 'salary', 'bonus', 'fraction_from_poi_email', 'fraction_to_poi_email', 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred']
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", "shared_receipt_with_poi"]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# print "We have ", len(data_dict.keys()), " executives in Enron Dataset."
# print len(data_dict['METTS MARK'].keys())
# print data_dict['METTS MARK'].keys()
# for key in data_dict.keys():
#     if data_dict[key]['poi']:
#         print data_dict[key]['email_address']
# print "---------------------------"


### Task 2: Remove outliers
# # Uncomment for see charts
# data = featureFormat(data_dict, ["salary", "bonus"])
# ### Assign data for chart
# for point in data:
#     plt.scatter(point[0], point[1])
#
# plt.xlabel("salary")
# plt.ylabel("bonus")
# plt.show()
# plt.clf()
#
# # Remove "Total" and check chart again
# data_dict.pop('TOTAL', 0)
# data = featureFormat(data_dict, ["salary", "bonus"])
# ### Re assign data for chart
# for point in data:
#     plt.scatter(point[0], point[1])
#
# plt.xlabel("salary")
# plt.ylabel("bonus")
# plt.show()
# plt.clf()


# outliers_dict = {}
# for employee_name in data_dict.keys():
#     count = 0
#     for key in data_dict[employee_name].keys():
#         if data_dict[employee_name][key] == 'NaN':
#             count += 1
#
#     if count in outliers_dict.keys():
#         outliers_dict[count] += [employee_name]
#     else:
#         outliers_dict[count] = [employee_name]
# print outliers_dict

outliers = ['TOTAL', "LOCKHART EUGENE E", 'THE TRAVEL AGENCY IN THE PARK']
for i in outliers:
    data_dict.pop(i, 0)

### Task 3: Create new feature(s)

# Calculate frations
fraction_from_poi_email = enron_tools.calculate_fraction(data_dict, "from_poi_to_this_person", "to_messages")
fraction_to_poi_email = enron_tools.calculate_fraction(data_dict, "from_this_person_to_poi", "from_messages")
# Add new feature values to data_dict
data_dict = enron_tools.add_features(data_dict, "fraction_from_poi_email", fraction_from_poi_email)
data_dict = enron_tools.add_features(data_dict, "fraction_to_poi_email", fraction_to_poi_email)

# Add new features to feature list
# features_list += ["fraction_from_poi_email", "fraction_to_poi_email"]

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.1, random_state = 42)

### Use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]


# DecisionTreeClassifier
t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred = clf.predict(features_test)
print 'Accuracy before tuning ', score
print 'Precision before tuning ', precision_score(labels_test, pred)
print 'Recall before tuning ', recall_score(labels_test,pred)
print "Decision tree algorithm time:", round(time()-t0, 3), "s"


print "\n\nValidating algorithm:"
### Manual tuning parameter min_samples_split
t0 = time()
clf = DecisionTreeClassifier(min_samples_split = 25)
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print 'Accuracy after tuning ', score
print 'Precision after tuning ', precision_score(labels_test, pred)
print 'Recall after tuning ', recall_score(labels_test,pred)
print "Decision tree algorithm time:", round(time()-t0, 3), "s"

# # See importances of features
# importances = clf.feature_importances_
# import numpy as np
# indices = np.argsort(importances)[::-1]
# print 'Feature Ranking: '
# print
# for i in range(len(clf.feature_importances_)):
#     print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])


# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators = 3)
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# score = clf.score(features_test, labels_test)
# print 'AdaBoostClassifier: ', score


# #GaussianNB
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# print "---"
# print "Accuracy: ", clf.score(features_test,labels_test), " Precision: ", precision_score(labels_test, pred), " Recall ",  recall_score(labels_test, pred)
# print "---"


# from time import time
# t0 = time()
#
# clf = GaussianNB()
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
# accuracy = accuracy_score(pred,labels_test)
# print "Accuracy: ", clf.score(features_test,labels_test), " Precision: ", precision_score(labels_test, pred), " Recall ",  recall_score(labels_test, pred)
#
# print "NB algorithm time:", round(time()-t0, 3), "s"

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)