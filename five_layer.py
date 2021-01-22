# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:15:08 2020

@author: Joshua
"""


from sklearn import metrics
import pandas as pd
import numpy as np
import time
import sklearn
from sklearn.model_selection import KFold, cross_validate
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_validate
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras import models
from keras.layers import Dense, Dropout, LSTM, Conv1D, Flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


data = './UNSW_NB15_training-set.csv' 

train_data = pd.read_csv(data)
det_labels = np.array(train_data['label'])
cat_features = pd.get_dummies(train_data['attack_cat'])
#cat_features = train_data['attack_cat']
train_cat_labels = np.array(cat_features)
#train_cat_list = list(cat_features.columns)
features= train_data.drop('label', axis = 1)
features= features.drop('id', axis = 1)
features= features.drop('attack_cat', axis = 1)
features = pd.get_dummies(features)
features_list = list(features.columns)
#print(features_list)
features = np.array(features)

idx = np.random.permutation(len(features))
x,y_det,y_cat = features[idx], det_labels[idx], train_cat_labels[idx]
d = 0.5
iterate = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
for i in iterate:
    print(i)
    model = models.Sequential()
    model.add(Dense(i, input_shape=(194,), activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(i, activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(i, activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(i, activation='relu'))
    model.add(Dropout(d))
    model.add(Dense(i, activation='relu'))
    model.add(Dropout(d))
    # model.add(Dense(i, activation='relu'))
    # model.add(Dropout(d))
    # model.add(Dense(i, activation='relu'))
    # model.add(Dropout(d))
    # model.add(Dense(i, activation='relu'))
    # model.add(Dropout(d))
    # model.add(Dense(i, activation='relu'))
    # model.add(Dropout(d))
    # model.add(Dense(i, activation='relu'))
    # model.add(Dropout(d))
    
    model.add(Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(lr=0.001)
    # Compile model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    Wsave = model.get_weights()
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state = 3)
    kf.get_n_splits(x)
    
    for train_index, test_index in kf.split(x):
        #print("TRAIN:", train_index, "TEST:", test_index)
        scaler = StandardScaler()
    
    
        X_train, X_test, = x[train_index], x[test_index]
        y_train, y_test = y_det[train_index], y_det[test_index]
    
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    
    
        model.set_weights(Wsave)
        model.fit(X_train, y_train,
              batch_size=128,
              epochs=200,
              verbose=2,
              validation_data=(X_test, y_test))



print('Thats All Folks!')













