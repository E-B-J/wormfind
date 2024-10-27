# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:47:42 2024

@author: ebjam
Training a random forest to select true worms
"""

import pickle, os
import pandas as pd
from scipy.stats import zscore
#%%
def get_annotated(path, wbw):
    with open(path, 'rb') as f:
        detections = pickle.load(f)
    df = pd.DataFrame(detections)
    good = wbw + "good/"
    bad = wbw + "bad/"
    good_images = [q[:-4] for q in os.listdir(good)]
    bad_images = [q[:-4] for q in os.listdir(bad)]
    all_images = good_images + bad_images
    df['training'] = 0
    df.loc[df['lookup'].isin(all_images), 'training'] = 1
    all_annotated = df.loc[df['training'] == 1].copy()
    all_annotated['label_str'] = 'bad'
    all_annotated.loc[all_annotated['lookup'].isin(good_images), 'label_str'] = 'good'
    training_features = ['area', 'perimeter', 'circularity', 'solidity', 'area_to_perimeter']
    for feature in training_features:
        all_annotated[f'zscore_{feature}'] = all_annotated.groupby('image')[feature].transform(lambda x: zscore(x, nan_policy='omit'))
    return(all_annotated)


wbw = "E:/toronto_microscopy/ixmc/sep_30_success/N2_lv4440_cyc1_curve/N2 lv4440 vs cyc1 2mil_Plate_2281/TimePoint_1/dapi/one_field/worm_by_worm/"
path = r"E:\toronto_microscopy\ixmc\sep_30_success\N2_lv4440_cyc1_curve\N2 lv4440 vs cyc1 2mil_Plate_2281\TimePoint_1\dapi\one_field\lv4440_vs_cyc1_2mil_curve_dapi_sato_unfiltered_oct3.pkl"

#%%

pkl_to_wbw = {
    r"E:\toronto_microscopy\ixmc\sep_30_success\N2_lv4440_cyc1_curve\N2 lv4440 vs cyc1 2mil_Plate_2281\TimePoint_1\dapi\one_field\lv4440_vs_cyc1_2mil_curve_dapi_sato_unfiltered_oct3.pkl": "E:/toronto_microscopy/ixmc/sep_30_success/N2_lv4440_cyc1_curve/N2 lv4440 vs cyc1 2mil_Plate_2281/TimePoint_1/dapi/one_field/worm_by_worm/",
    r"E:\toronto_microscopy\ixmc\Oct_8_ALB & N2vsAWR73\Oct 08 ALB RT_Plate_2301\TimePoint_1\dapi\one_field\n2_alb_rep1_dapi_sato_unfiltered_oct21.pkl": "E:/toronto_microscopy/ixmc/Oct_8_ALB & N2vsAWR73/Oct 08 ALB RT_Plate_2301/TimePoint_1/dapi/one_field/worm_by_worm/",
    r"E:\toronto_microscopy\ixmc\Oct_8_ALB & N2vsAWR73\Oct 08 N2VAWR73 RT_Plate_2302\TimePoint_1\dapi\one_field\n2_v_awr73_rep1_dapi_sato_unfiltered_oct21.pkl": "E:/toronto_microscopy/ixmc/Oct_8_ALB & N2vsAWR73/Oct 08 N2VAWR73 RT_Plate_2302/TimePoint_1/dapi/one_field/worm_by_worm/"
}

# Initialize an empty list to collect annotated DataFrames
annotated_dfs = []

# Loop through the dictionary items
for key, value in pkl_to_wbw.items():
    anno = get_annotated(key, value)
    annotated_dfs.append(anno)

# Concatenate all annotated DataFrames into one
all_annotated = pd.concat(annotated_dfs, ignore_index=True)

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

to_train_with = all_annotated[['circularity', 'perimeter', 'solidity', 'area']].copy()
#to_train_with = all_annotated.drop(['image', 'lookup', 'bbox', 'contour', 'dy96_pixels', 'dy96_percent', 'fish_pixels', 'fish_percent', 'approximate_contour', 'training', 'label_str'], axis=1).copy()
to_train_with.fillna(0, inplace=True)
y = all_annotated['label_str'].copy()
X_train, X_test, y_train, y_test = train_test_split(to_train_with, y, test_size = 0.4, random_state = 959, stratify = y) 
param_grid = { 
    'n_estimators': [100, 150, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [1, 2, 5, 8, 10, 40],
    'min_samples_leaf': [1, 4, 8, 16, 32],
    'max_depth' : [10, 20, 30, None],
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

#%%
import numpy as np
import matplotlib.pyplot as plt

# Show feature importances
importances = best_rf.feature_importances_
feature_names = to_train_with.columns
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

#%%
import joblib

joblib.dump(best_rf, "C:/Users/ebjam/Documents/GitHub/wormfind/models/worm_detection_classifier/best_sato_Oct24_noZscore.joblib")