#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
#################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# from sklearn.neighbors import KNeighborsRegressor
# clf = KNeighborsRegressor(n_neighbors=20)
# clf.fit(features_train, labels_train)
#
# print "KNeighbors"
# print(clf.score(features_test, labels_test))
# Default => 0.748995983936
# n_neighbors=5 => 0.784853700516
# n_neighbors=10 => 0.810671256454
# n_neighbors=20 => 0.820308017785



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=5, min_samples_split=40)
clf.fit(features_train, labels_train)
print(clf.score(features_test, labels_test))
# # n_estimators=10, min_samples_split=40 => 0.924
# # n_estimators=5, min_samples_split=40 => 0.936


# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(algorithm='SAMME')
# clf.fit(features_train, labels_train)
# print "AdaBoostClassifier"
# print(clf.score(features_test, labels_test))
# Dafault => 0.924
# n_estimators=25 => 0.924



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
