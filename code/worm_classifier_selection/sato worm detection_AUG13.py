# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 09:17:09 2024

@author: ebjam
"""

import os
import cv2
import numpy as np
from skimage import io
from skimage.filters import frangi, sato
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
#%%
# Define the folder containing the images
input_folder = 'E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/dapi/one_field/'
dy96_folder = 'E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/dy96/one_field/'
fish_folder = 'E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/fish/one_field/'
output_folder = input_folder + "vesselness/"
worm_by_worm = input_folder + "worm_by_worm/"
d_worm_by_worm = dy96_folder + "worm_by_worm/"
f_worm_by_worm = fish_folder + "worm_by_worm/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(worm_by_worm, exist_ok=True)
os.makedirs(d_worm_by_worm, exist_ok=True)
os.makedirs(f_worm_by_worm, exist_ok=True)
detections = []
figure_size = (10,10)
figure_dpi = 100
for filename in os.listdir(input_folder):
    if filename.endswith("TIF"):  
        print(filename)
        file_path = os.path.join(input_folder, filename)
        dy_filename = filename.replace("w1_combo", "w2_combo")
        fish_filename = filename.replace("w1_combo", "w3_combo")
        image = cv2.imread(file_path, -1)
        dimage = cv2.imread(dy96_folder + dy_filename, -1)
        fimage = cv2.imread(fish_folder + fish_filename, -1)
        # Apply the Sato Tubeness - ## We're pretending each worm is a tiny bloodvessel, don't let the code know they're actually worms
        # Ahem, Take a look at these blood vessels computer! Aren''t they cool?
        svesselness = sato(image, black_ridges=False)
        svesselness_ubyte = img_as_ubyte(svesselness)
        svesselness_ubyte = cv2.blur(svesselness_ubyte, (3,3))
        _, sv_thresh = cv2.threshold(svesselness_ubyte, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(sv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(image, contours, -1, color, thickness=cv2.FILLED)
        #fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
        #ax.imshow(image)
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
                mask = np.zeros(image.shape, dtype=np.uint16)
                cv2.drawContours(mask, [contour], -1, 65535, thickness=cv2.FILLED)  # Fill the contour
                # Extract the pixel values within the contour
                masked_image = cv2.bitwise_and(image, mask)
                masked_dimage = cv2.bitwise_and(dimage, mask)
                masked_fimage = cv2.bitwise_and(fimage, mask)
                cropped_dapi = masked_image[y:y+h, x:x+w]
                cropped_masked_dimage = masked_dimage[y:y+h, x:x+w]
                cropped_masked_fimage = masked_fimage[y:y+h, x:x+w]
                num_fish_pixels = len(cropped_masked_fimage[cropped_masked_fimage > 13000])
                #plt.imshow(cropped_masked_fimage)
                #plt.show()
                pixel_values = cropped_dapi[cropped_dapi > 0]
                # Get only non-zero pixel values
                mean_value = np.mean(pixel_values)
                print(mean_value)
                max_val = max(pixel_values)
                min_val = min(pixel_values)
                measurements['mean'] = mean_value
                measurements['max'] = max_val
                measurements['min'] = min_val
                measurements['dapi_std'] = np.std(pixel_values)
                measurements['contour'] = contour
                measurements['fish_pixels'] = num_fish_pixels
                measurements['fish_percent'] = 100 * (num_fish_pixels / area)
                detections.append(measurements)
                namer += 1

                cropped_image = image[y:y+h, x:x+w]
                cropped_dy96 = dimage[y:y+h, x:x+w]
                cv2.imwrite(worm_by_worm + lookup + ".png", cropped_image)
                cv2.imwrite(d_worm_by_worm + lookup + "dy96.png", cropped_dy96)
                cv2.imwrite(f_worm_by_worm + lookup + "fish.png", cropped_masked_fimage)
import gzip, pickle
with gzip.open(worm_by_worm + 'aug3_n2_vs_ju1400_unfiltered_detections.pkl.gz', 'wb') as f:
    pickle.dump(detections, f)
print("Saved!")

#%%
import pandas as pd

df = pd.DataFrame(detections)
df['single_class_target'] = 0
#%%

true_worms = [q[:-4] for q in os.listdir("E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/dapi/one_field/worm_by_worm/first_run/worms/")]
well_artifacts = [q[:-4] for q in os.listdir("E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/dapi/one_field/worm_by_worm/first_run/well_artifacts/")]
embryos_and_debris = [q[:-4] for q in os.listdir("E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/dapi/one_field/worm_by_worm/first_run/debris_and_embryos/")]
clusters = [q[:-4] for q in os.listdir("E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/dapi/one_field/worm_by_worm/first_run/clusters/")]
l1 = [q[:-4] for q in os.listdir("E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/dapi/one_field/worm_by_worm/first_run/L1/")]
#%%
def update_single_target(row):
    if row['lookup'] in true_worms:
        return 1
    return row['single_class_target']

def update_multiclass(row):
    if row['lookup'] in well_artifacts:
        return 2
    elif row['lookup'] in embryos_and_debris:
        return 3
    elif row['lookup'] in clusters:
        return 4
    elif row['lookup'] in l1:
        return 5
    return row['multiclass_target']
# Apply the custom function to update the 'Score' column
df['single_class_target'] = df.apply(update_single_target, axis=1)
df['multiclass_target'] = df['single_class_target'].copy()
df['multiclass_target'] = df.apply(update_multiclass, axis=1)


#%%

sns.regplot(x='area', y = 'single_class_target', data=df, order = 2)

sns.regplot(x='perimeter', y = 'single_class_target', data=df, order = 2)

sns.regplot(x='circularity', y = 'single_class_target', data=df, order = 2)

sns.regplot(x='solidity', y = 'single_class_target', data=df, order = 2)

sns.regplot(x='dapi_std', y = 'single_class_target', data=df, order = 2)

variables_for_random_forest = ['area', 'perimeter', 'circularity', 'solidity', 'mean', 'dapi_std', 'area_to_perimeter']

#%%
# make derivitives

df['area_to_perimeter'] = df['area'] / df['perimeter']
sns.regplot(x='area_to_perimeter', y = 'single_class_target', data=df, order = 2)
#%%
for image in np.unique(df['image']):
    rele_well = df.loc[df['image'] == image]
    for feature in variables_for_random_forest:
        feature_list = rele_well[feature].tolist()
        mean = np.mean(feature_list)
        std = np.std(feature_list)
        df.loc[df['image'] == image, feature + '_well_zscore'] = (rele_well[feature] - mean) / std
        #rele_well[feature + '_well_zscore'] = (rele_well[feature] - mean) / std
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler

#%%
z_scores = [q + '_well_zscore' for q in variables_for_random_forest] 
full_forest = variables_for_random_forest + z_scores
# No z scores right now!!
X = df[full_forest]
y = df['single_class_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=959)

param_grid = {
    'n_estimators': [50, 100, 115, 150, 200],
    'max_depth': [None, 10,  15,  20,  25],
    'min_samples_split': [1, 2, 5, 7, 10, 25],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=959)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
#%%

print(f"Best parameters found: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


#%%

with gzip.open(worm_by_worm + 'aug3_n2_vs_ju1400_detections_used_to_train.pkl.gz', 'wb') as f:
    pickle.dump(df, f)
print("Saved!")
#%%
import joblib

joblib.dump(best_rf, worm_by_worm + 'sato_worm_detection_predictor.joblib')