#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

print features_train

clf = SVC(kernel='rbf', C=10000)
# => C=10.0:    0.616040955631
# => C=100.0:   0.616040955631
# => C=1000.0:  0.821387940842
# => C=10000.0: 0.892491467577

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
# => RBF: 0.138 s

t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
# => RBF: 1.478 s


#print(clf.score(features_test, labels_test))

sara = 0
chris = 0
for value in pred:
    if value == 0:
        sara += 1
    elif value == 1:
        chris += 1

print "Sara => ", sara
print "Chris => ", chris

accur = accuracy_score(pred, labels_test)

print("Accuracy")
# RBF: 0.616040955631
# RBF,C=10000.0,full dataset: 0.990898748578
print(accur)

#########################################################


# print clf.predict(features_test[10])
# print clf.predict(features_test[26])
# print clf.predict(features_test[50])