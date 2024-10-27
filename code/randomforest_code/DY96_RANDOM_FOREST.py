# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:55:55 2024

@author: ebjam

dy96 random forest id - same as FISH really...
"""

import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from skimage import feature, future, color
import pandas as pd

#%%
'''
background = 1
worm = 2
embryo = 3
spore = 4
'''
#%%

trainable_pixels = pd.DataFrame()
image_path = "E:/toronto_microscopy/ixmc/sep_30_success/training_dy96_segmentation/noclahe/"
label_path = "E:/toronto_microscopy/ixmc/sep_30_success/training_dy96_segmentation/noclahe/labels"
for image in os.listdir(label_path):
    if image.endswith("png"):
        label_img = cv2.imread(os.path.join(label_path, image), -1)
        if np.all(label_img == 0):
            continue
        clahe_handle = image.replace("_Labels", "")[:-4] + ".TIF"
        
        clahe_img = cv2.imread(os.path.join(image_path, clahe_handle), -1)
        
        sig_min = 1
        sig_max = 16
        features_func = partial(
            feature.multiscale_basic_features,
            intensity = True,
            edges = True,
            texture = True,
            sigma_min = sig_min,
            sigma_max = sig_max,
            )
        features = features_func(clahe_img)
        X = features.reshape(-1, features.shape[-1])
        y = label_img.ravel()
        label_df = pd.DataFrame(y)
        df_features = pd.DataFrame(X)
        df_features['image'] = image
        df_features['label'] = label_df
        # I want to hold on to the labelled stuff and keep it to train a random forest on!
        trainable_area = df_features.loc[df_features['label'] != 0].copy()
        trainable_pixels = pd.concat([trainable_pixels, trainable_area], ignore_index = True)
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

to_train_with = trainable_pixels.drop(['image', 'label'], axis=1).copy()
y = trainable_pixels['label'].copy()
X_train, X_test, y_train, y_test = train_test_split(to_train_with, y, test_size = 0.4, random_state = 959) 
param_grid = { 
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 20],
    'min_samples_leaf': [1, 8, 32],
    'max_depth' : [5, 10, 20, None],
    'criterion' :['gini', 'entropy'],
    'bootstrap' : ['True', 'False']
}

clf = GridSearchCV(RandomForestClassifier(random_state=959), param_grid, cv=5, verbose=True, n_jobs=-1)

clf.fit(X_train, y_train)

best_rf = clf.best_estimator_
best_params = clf.best_params_
best_score = clf.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
X_test['label'] = y_test
just_spore = X_test.loc[X_test['label'] == 4].copy()

js_x = just_spore.drop(['label'], axis=1).copy()
js_y = just_spore['label'].copy()


accuracy = clf.score(js_x, js_y)

#%%
print('Spore Accuracy = ' + str(accuracy))



#%%
import joblib
joblib.dump(best_rf, "E:/toronto_microscopy/ixmc/sep_30_success/training_dy96_segmentation/noclahe/best_RF_DY96_acc=_Oct24.joblib")