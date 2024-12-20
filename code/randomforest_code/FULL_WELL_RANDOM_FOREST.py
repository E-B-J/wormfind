# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:17:10 2024

@author: ebjam
"""

import os, cv2
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from skimage import feature
import pandas as pd
from matplotlib.patches import Polygon

# Frame = 1
# Background = 2
# Worm = 3
# Debris = 4
# Embryo = 5

trainable_pixels = pd.DataFrame()
image_path = "E:/toronto_microscopy/ixmc/Sep16_detergent_tests/p1_e_f_independent/TimePoint_1/dapi/one_field/resize_for_rf/"
label_path = "E:/toronto_microscopy/ixmc/Sep16_detergent_tests/p1_e_f_independent/TimePoint_1/dapi/one_field/resize_for_rf/labels/"
for image in os.listdir(label_path):
    if image.endswith("png"):
        label_img = cv2.imread(os.path.join(label_path, image), -1)
        if np.all(label_img == 0):
            continue
        onefield_handle = image.replace("_Labels", "")
        onefield_handle = onefield_handle.replace("png", "TIF")
        onefield_img_raw = cv2.imread(os.path.join(image_path, onefield_handle), -1)
        onefield_img = cv2.resize(onefield_img_raw, (1024, 1024))
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
        features = features_func(onefield_img)
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
X_train, X_test, y_train, y_test = train_test_split(to_train_with, y, test_size = 0.4, random_state = 959, stratify = y) 
param_grid = { 
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 10, 40],
    'min_samples_leaf': [1, 8, 32],
    'max_depth' : [10, 20, None],
    'criterion' :['gini', 'entropy'],
    'bootstrap' : ['True', 'False']
}

clf = GridSearchCV(RandomForestClassifier(random_state=959), param_grid, cv=5, verbose=True, n_jobs=-1)

clf.fit(X_train, y_train)
#%%

best_rf = clf.best_estimator_
best_params = clf.best_params_
best_score = clf.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")
#%%
import joblib
joblib.dump(best_rf, "E:/toronto_microscopy/ixmc/Sep16_detergent_tests/p1_e_f_independent/best_onefield_segmenter_sep18.joblib")
#%%
X_test['label'] = y_test
#%%
just_spore = X_test.loc[X_test['label'] == 3].copy()

js_x = just_spore.drop(['label'], axis=1).copy()
js_y = just_spore['label'].copy()


accuracy = clf.score(js_x, js_y)
#%%
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
#%%
dapi_ones = [q for q in os.listdir(image_path)]

features_func = partial(
    feature.multiscale_basic_features,
    intensity = True,
    edges = True,
    texture = True,
    sigma_min = sig_min,
    sigma_max = sig_max,
    )
detections = []
for filename in dapi_ones:
    img = cv2.imread(os.path.join(image_path, filename), -1)
    img_holder = cv2.imread(os.path.join(image_path, filename), -1)
    img_small = cv2.resize(img, (1024, 1024))
    features = features_func(img_small)
    X = features.reshape(-1, features.shape[-1])
    df_features = pd.DataFrame(X)
    worm_prediction = best_rf.predict(df_features)
    worm_prediction_img = worm_prediction.reshape((1024,1024))
    worm_prediction_img = cv2.resize(worm_prediction_img, (4096, 4096), interpolation = cv2.INTER_NEAREST)
    just_worms = np.where(worm_prediction_img == 3, 255, 0).astype(np.uint8)
    plt.imshow(just_worms)
    plt.show()
    contours, _ = cv2.findContours(just_worms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    empty_image = np.zeros((4096, 4096), dtype=np.uint8)

# Draw contours on the empty image
    cv2.drawContours(empty_image, contours, -1, (255), thickness=cv2.FILLED)
# Display the contours on the empty image
    plt.imshow(empty_image, cmap='gray')
    plt.title('Contours')
    plt.axis('off')
    plt.show()
    img_holder = cv2.resize(img, (4096, 4096))
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        x, y, w, h = bbox
        crop = img_holder[y:y+h, x:x+w]
        plt.imshow(crop)
        plt.show()
    
    #%%
    namer = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            measurements = {}
            measurements['image'] = filename
            lookup = filename + "_worm_" + str(namer)
            measurements['lookup'] = lookup
            color = [q/255 for q in list(np.random.choice(range(255), size=3))]
            polygon = Polygon(contour[:, 0, :], closed=True, fill=True, fc=color, ec='k', alpha=1)
            #ax.add_patch(polygon)
            measurements['area'] = area
            perimeter = cv2.arcLength(contour, True)
            measurements['perimeter'] = perimeter

            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            measurements['circularity'] = circularity
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            measurements['solidity'] = solidity
            bbox = cv2.boundingRect(contour)
            measurements['bbox'] = bbox
            x, y, w, h = bbox
            crop = img[y:y+h, x:x+w]
            plt.imshow(crop)
            plt.show()
            measurements['contour'] = contour
            detections.append(measurements)
            namer += 1
    plt.show()

#%%
big_bbox = [q*4 for q in bbox]
big_contour = contour * 4
#%%
target_value = 3
filtered_image = np.where(worm_prediction_img == target_value, 255, 0)
plt.imshow(filtered_image)
plt.show()


