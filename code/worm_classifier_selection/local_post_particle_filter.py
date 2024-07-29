# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:21:47 2024

@author: ebjam
"""

import pickle
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import sklearn
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%%
load_path = "E:/Toronto_microscopy/June12-N2-Double-Curve_Plate_2061/June12-N2-Double-Curve_Plate_2061/TimePoint_1/dapi/one_field/July_9_while_plate_unfiltered_worms.pkl"

with open(load_path, 'rb') as f:
    loaded_list = pickle.load(f)
    
df_full = pd.DataFrame(loaded_list)
selection_for_forest_filter = ['Area', 'Mean', 'Mode', 'Min', 'Max', 'Perim.', 'Circ.', 'Feret', 'Round', 'Solidity']

path_to_rf_predictor = "C:/Users/ebjam/July_4_best_rf_broad_search.joblib"
rf_predictor = joblib.load(path_to_rf_predictor)

def predict_on_row(row):
    row_features = pd.DataFrame([row[selection_for_forest_filter]])
    return rf_predictor.predict(row_features)[0]

df_full['prediction'] = df_full.apply(predict_on_row, axis=1)
#%%
true_worms = df_full.loc[df_full['prediction'] == 1].copy()
false_worms = df_full.loc[df_full['prediction'] == 0]

true_worms['well'] = true_worms['image'].str[23:26]
true_worms['row'] = true_worms['well'].str[0]
true_worms['col'] = true_worms['well'].str[1:]
true_worms['plate'] = true_worms['image'].str[:16]
#%%
true_worm_save_path = "E:/Toronto_microscopy/June12-N2-Double-Curve_Plate_2061/June12-N2-Double-Curve_Plate_2061/TimePoint_1/dapi/one_field/test/whole_plate_true_worms.pkl"
with open(true_worm_save_path, 'wb') as f:
    pickle.dump(true_worms, f)
    
    
    
#%%
#  To visualize difference, would be nice to plot both good worms and bad worms ontop of input image to show that filtering matters...

images = true_worms['image'].unique()
image_path = "E:/Toronto_microscopy/June12-N2-Double-Curve_Plate_2061/June12-N2-Double-Curve_Plate_2061/TimePoint_1/dapi/one_field/"
for image in images:
    print(image)
    img_segmentations = []
    relevant_detections = true_worms.loc[true_worms['image'] == image]
    for index, row in relevant_detections.iterrows():
        raw_segmentation = row['segmentation']
        img_segmentations.append(raw_segmentation)
    # Load image
    img_path = os.path.join(image_path, image)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    # Make all segs into polygons
    for seg in img_segmentations:
        random_color = np.random.rand(3)
        random_color = np.append(random_color, 1)
    # Close polygon if necessary
        if seg[0] != seg[-1]:
            seg.append(seg[-1])
    # Plot all segs onto original image
        patch = patches.Polygon(seg, closed=True, color=random_color, ec='white', lw=1)
        ax.add_patch(patch)
    # Print image, and move onto next
    plt.title(image)
    plt.axis('off')
    plt.show()
    
    
    
    
    