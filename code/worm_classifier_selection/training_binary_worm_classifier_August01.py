# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:01:05 2024

@author: ebjam
"""

"""
Created on Mon Jul 29 07:59:13 2024

@author: ebjam
"""
import pandas as pd
import os, cv2, gzip, pickle
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
#%%

load_path = r"E:\toronto_microscopy\ixmc\July28LiquidlotRNAi\July28LiquidlotRNAi\TimePoint_1\dapi\one_field\binary\Liquid_lot_RNAI_unfiltered_worms.gz"

with gzip.open(load_path, 'rb') as f:
    loaded_list = pickle.load(f)
    
df = pd.DataFrame(loaded_list)
images = df['image'].unique()
image_path = "E:/toronto_microscopy/ixmc/July28LiquidlotRNAi/July28LiquidlotRNAi/TimePoint_1/dapi/one_field/binary/"
worm_by_worm = image_path + 'worm_by_worm/'
os.makedirs(worm_by_worm, exist_ok=True)
for image in images:
    #tif_image = image.replace('.png', '.TIF')
    dapi_img = cv2.imread(os.path.join(image_path, image))
    print(image)
    relevant_detections = df.loc[df['image'] == image]
    for index, row in relevant_detections.iterrows():
        # Fiji gives decimals, now we need to operate at pixels instead of fractions of pixels.
        # To make sure I'm not cropping any of the segmentation out, I'm rounding down x & y, and rounding up H & W to nearest int
        x = math.floor(row['BX']) 
        y = math.floor(row['BY'])
        height = math.ceil(row['Height'])
        width = math.ceil(row['Width'])
        name = row['lookup'].replace(".TIF", "")
        name += ".png"
        this_worm = dapi_img[y:y+height, x:x+width]
        cv2.imwrite(os.path.join(worm_by_worm, name), this_worm)
#%%
# Break here to sort all detections into good and bad folders - will uses these to make target!
#%%
#%%
bad = [q[:-4] for q in os.listdir("E:/toronto_microscopy/ixmc/July28LiquidlotRNAi/July28LiquidlotRNAi/TimePoint_1/dapi/one_field/binary/worm_by_worm/bad/")]
#bad_adjust = [q.replace("bo_", "bo.TIF_") for q in bad]
#bad_adjust_2 = [q.replace(".png", "") for q in bad_adjust]
found_in_bad = [q for q in bad if q in df['lookup'].unique()]
df['target'] = 1
df.loc[df['lookup'].isin(found_in_bad), 'target'] = 0
df['perimeter:area'] = df['Perim.'] / df['Area']
# Took a look at a few plots, I can filter the bottom and top on area and barely remove any real detections.
area_filtered_df = df.loc[df['Area'] > 1000].copy()
area_filtered_df = df.loc[df['Area'] < 32000].copy()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_df =  area_filtered_df.select_dtypes(include=numerics)
y = train_df['target']
X = train_df.drop(columns = ['target', 'Mean', 'Max', 'Median', 'Mode', 'BX', 'BY', 'Height', 'Width', 'FeretAngle'])
#%%
#lets train a classifier...



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Predict on the test set using the best estimator
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)

#%%
feature_importances = best_rf.feature_importances_
feature_names = X.columns

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(importance_df)

# Optionally, plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()
#%%
# Great! We can see that min, FeretX, FeretY are all below 0.01 for importance, we'll drop them and redo



X_important = X.drop(columns = ['Min', 'FeretX', 'FeretY']) 

X_train, X_test, y_train, y_test = train_test_split(X_important, y, test_size=0.4, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [1, 2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, refit = 'roc_auc',
                           cv=5, n_jobs=-1, verbose=2, scoring=scoring)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Predict on the test set using the best estimator
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)


# Print metrics
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC-AUC: {roc_auc}')

#%%
# Excellent! We're up at about Accuracy = 0.93! Lovely jubbly!
# Lets save this model for later


joblib.dump(best_rf, 'C:/Users/ebjam/Documents/GitHub/wormfind/models/worm_detection_classifier/best_random_forest_binary_august01.joblib')