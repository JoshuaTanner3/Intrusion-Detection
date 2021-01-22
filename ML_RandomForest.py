# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 07:56:08 2020

@author: Joshua
"""


from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import sklearn

train_data = pd.read_csv('./../UNSW_NB15_training-set.csv')
#test_data = pd.read_csv('./UNSW_NB15_testing-set.csv')

#-----------------------------------------------------------train data process
det_labels = np.array(train_data['label'])
cat_features = pd.get_dummies(train_data['attack_cat'])
cat_features = train_data['attack_cat']
train_cat_labels = np.array(cat_features)
#train_cat_list = list(cat_features.columns)
features= train_data.drop('label', axis = 1)
#features= train_data.drop('id', axis = 1)
features= features.drop('attack_cat', axis = 1)
features= features.drop('id', axis = 1)
features = pd.get_dummies(features)
features_list = list(features.columns)
features = np.array(features)

from sklearn.preprocessing import StandardScaler
scaling = StandardScaler().fit(features)
features = scaling.transform(features)

idx = np.random.permutation(len(features))
x,y = features[idx], det_labels[idx]

#------------------------------------------------------------create and train detection model
#train_features, test_features, train_labels, test_labels = train_test_split(features, det_labels, test_size = 0.1, random_state = 3)
iterate = [2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
det_scores = []
for i in iterate:
    print('Det', i)
    clf_det=RandomForestClassifier(n_estimators=i)
    # Perform 7-fold cross validation 
    scores = sklearn.model_selection.cross_val_score(estimator=clf_det, X=x, y=y, cv=5)
    for score in scores:
        det_scores.append([i, score])
        print(score)

# n_splits = 5
# kf = KFold(n_splits=n_splits, shuffle=True, random_state = 3)
# kf.get_n_splits(features)


# for train_index, test_index in kf.split(features):
#     #print("TRAIN:", train_index, "TEST:", test_index)

#     X_train, X_test = x[train_index], x[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     clf=RandomForestClassifier(n_estimators=5)
#     clf.fit(X_train,y_train) 
#     y_pred=clf.predict(X_test)
#     print("Random Forest Detection Accuracy:",metrics.accuracy_score(y_pred, y_test))


print(det_scores)

column_name = ['n_estimators', 'acc']
final_df = pd.DataFrame(det_scores, columns=column_name)
        
final_df.to_csv('./Random_Forest_GOOD_Det_Results.csv', index=None)
print('Successfully converted xml to csv.')


# clf_det.fit(train_features,train_labels)
# y_pred_det=clf_det.predict(test_features)

#print("Random Forest Detection Accuracy:",metrics.accuracy_score(test_labels, y_pred_det))


#-------------------------------------------------------------create and train classification model
#train_features, test_features, train_labels, test_labels = train_test_split(features, train_cat_labels, test_size = 0.1, random_state = 3)

idx = np.random.permutation(len(features))
x,y = features[idx], train_cat_labels[idx]

#------------------------------------------------------------create and train detection model
#train_features, test_features, train_labels, test_labels = train_test_split(features, det_labels, test_size = 0.1, random_state = 3)
iterate = [2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200]
det_scores = []
for i in iterate:
    print('Cat', i)
    clf_det=RandomForestClassifier(n_estimators=i)
    # Perform 7-fold cross validation 
    scores = sklearn.model_selection.cross_val_score(estimator=clf_det, X=x, y=y, cv=5)
    for score in scores:
        det_scores.append([i, score])

print(det_scores)

column_name = ['n_estimators', 'acc']
final_df = pd.DataFrame(det_scores, columns=column_name)
        
final_df.to_csv('./Random_Forest_GOOD_Cat_Results.csv', index=None)
print('Successfully converted xml to csv.')





