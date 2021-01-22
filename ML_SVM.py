# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 07:56:08 2020

@author: Joshua
"""


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import numpy as np
import time
import sklearn


start = time.time()
train_data = pd.read_csv('./UNSW_NB15_training-set.csv')
#test_data = pd.read_csv('./UNSW_NB15_testing-set.csv')

#-----------------------------------------------------------train data process
det_labels = np.array(train_data['label'])
cat_features = pd.get_dummies(train_data['attack_cat'])
cat_features = train_data['attack_cat']
train_cat_labels = np.array(cat_features)
#train_cat_list = list(cat_features.columns)
features= train_data.drop('label', axis = 1)
features= features.drop('attack_cat', axis = 1)
features= features.drop('id', axis = 1)
features = pd.get_dummies(features)
features_list = list(features.columns)
features = np.array(features)


from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(features)
features = scaling.transform(features)


#------------------------------------------------------------create and train detection model
#train_features, test_features, train_labels, test_labels = train_test_split(features, det_labels, test_size = 0.1, random_state = 3)
print('before')
#clf_det=SVC(kernel='rbf', probability=False, cache_size=7000)



idx = np.random.permutation(len(features))
x,y = features[idx], det_labels[idx]

#------------------------------------------------------------create and train detection model
#train_features, test_features, train_labels, test_labels = train_test_split(features, det_labels, test_size = 0.1, random_state = 3)
iterate = ['rbf']
det_scores = []
for i in iterate:
    print('Det', i)
    clf_det=SVC(kernel=i, probability=False, cache_size=7000)
    # Perform 7-fold cross validation 
    scores = sklearn.model_selection.cross_val_score(estimator=clf_det, X=x, y=y, cv=5)
    for score in scores:
        det_scores.append([i, score])
        print(score)

print(det_scores)

# column_name = ['type', 'acc']
# final_df = pd.DataFrame(det_scores, columns=column_name)
        
# final_df.to_csv('./SVM_Det_Results.csv', index=None)
# print('Successfully converted xml to csv.')


# idx = np.random.permutation(len(features))
# x,y = features[idx], train_cat_labels[idx]



# iterate = ['rbf', 'linear']
# det_scores = []
# for i in iterate:
#     print('cat', i)
#     clf_det=SVC(kernel=i, probability=False, cache_size=7000)
#     # Perform 7-fold cross validation 
#     scores = sklearn.model_selection.cross_val_score(estimator=clf_det, X=x, y=y, cv=5)
#     for score in scores:
#         det_scores.append([i, score])

# print(det_scores)

# column_name = ['type', 'acc']
# final_df = pd.DataFrame(det_scores, columns=column_name)
        
# final_df.to_csv('./SVM_Cat_Results.csv', index=None)
# print('Successfully converted xml to csv.')









# print(end - start)

# #-------------------------------------------------------------create and train classification model
# #train_features, test_features, train_labels, test_labels = train_test_split(features, train_cat_labels, test_size = 0.1, random_state = 3)

# clf_cat=SVC(kernel='rbf', probability=False, cache_size=7000)

# clf_cat.fit(train_features,train_labels)
# y_pred_cat=clf_cat.predict(test_features)

# print("SVM Classification Accuracy:",metrics.accuracy_score(test_labels,y_pred_cat))
# end = time.time()
# print(end - start)




