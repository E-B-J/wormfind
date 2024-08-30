# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:03:36 2024

@author: ebjam
"""

import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from skimage import feature, future, color
import pandas as pd

#255 = void
#254 = background
#253 = spore
#252 = worm
value_mapping = {255:1, 254:2, 253:4, 252:3}
#%%
trainable_pixels = pd.DataFrame()
image_path = "E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/fish/one_field/worm_by_worm/clahe/"
label_path = "E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/fish/one_field/worm_by_worm/clahe/labels/"
for image in os.listdir(label_path):
    if image.endswith("png"):
        label_img = cv2.imread(os.path.join(label_path, image), -1)
        if np.all(label_img == 0):
            continue
        clahe_handle = image.replace("_Labels", "")
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
    'n_estimators': [50, 100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10, 20, 40],
    'min_samples_leaf': [1, 2, 4, 8, 16, 32],
    'max_depth' : [5, 10, 15, 20, 30, 40, 50, None],
    'criterion' :['gini', 'entropy'],
    'bootstrap' : ['True', 'False']
}

clf = GridSearchCV(RandomForestClassifier(random_state=959), param_grid, cv=5, verbose=True, n_jobs=-1)

clf.fit(X_train, y_train)


accuracy = clf.score(X_test, y_test)
#%%
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
#%%


clf = RandomForestClassifier(n_estimators)