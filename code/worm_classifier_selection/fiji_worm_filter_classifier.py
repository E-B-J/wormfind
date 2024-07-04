# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:07:03 2024

@author: ebjam
"""
# Data Processing
import pandas as pd
import numpy as np
import os
import pickle
import joblib

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import RobustScaler
# Vis
import matplotlib.pyplot as plt
import seaborn as sns
#%%

# Data prep:
# Have files in either a 'good' or 'bad' folder, using those to generate labels
# Have all features, using those to discriminate between good and bad detections
# Need to join labels and features, bare in mind, roi names can sometimes dupe - join on well + roi
good_path = r"E:/Toronto_microscopy/June12-N2-Double-Curve_Plate_2061/June12-N2-Double-Curve_Plate_2061/TimePoint_1/dapi/one_field/edge_detection/good/"
bad_path = r"E:/Toronto_microscopy/June12-N2-Double-Curve_Plate_2061/June12-N2-Double-Curve_Plate_2061/TimePoint_1/dapi/one_field/edge_detection/bad/"

good_images = [q for q in os.listdir(good_path) if q.endswith("png")]
bad_images = [q for q in os.listdir(bad_path) if q.endswith("png")]

def list_to_df(list_of_files, label):
    df = pd.DataFrame(list_of_files, columns=['handle'])
    df['label'] = label
    df['well'] = df['handle'].str[23:26]
    df['col'] = df['well'].str[1:]
    df['row'] = df['well'].str[0]
    return(df)

gdf = list_to_df(good_images, 'good')
gdf['target_label'] = 1
bdf = list_to_df(bad_images, 'bad')
bdf['target_label'] = 0
combo = pd.concat([gdf, bdf], ignore_index=True)
combo['roi'] = combo['handle'].str[-13:-4]
combo['lookup'] = combo['well'] +"_"+ combo['roi']


feature_data = pd.read_csv(r"E:\Toronto_microscopy\June12-N2-Double-Curve_Plate_2061\June12-N2-Double-Curve_Plate_2061\TimePoint_1\dapi\one_field\erode_sharpen_detect_worm_id_res_final.csv")
feature_data['roi'] = feature_data['Label'].str[-9:]
feature_data['well'] = feature_data['Label'].str[23:26]
feature_data['row'] = feature_data['well'].str[0]
feature_data['col'] = feature_data['well'].str[-2:]
feature_data['lookup'] = feature_data['well'] +"_"+ feature_data['roi']

feature_combo = combo.merge(feature_data, on='lookup')

#%%
import pickle
feature_combo_dump_path = "E:/Toronto_microscopy/June12-N2-Double-Curve_Plate_2061/June12-N2-Double-Curve_Plate_2061/TimePoint_1/dapi/one_field/test/"
with open("E:/Toronto_microscopy/June12-N2-Double-Curve_Plate_2061/June12-N2-Double-Curve_Plate_2061/TimePoint_1/dapi/one_field/test/combo_dump.pkl", 'wb') as pickle_dump:
    pickle.dump(feature_combo, pickle_dump)
#%%

# Now I want to calculate some within-well z scores.
# The assumption I'm making is that the majority of detections in a well area
# correct, and detections that re markedly different from others within their well are invalid
# Therefore z scores should be informative in discriminating 'good' from 'bad' detections
# adding a within-well measure is pleasing to me, as it should help to not
# remove unusual phenotypes as debris.

wells = feature_combo['well_x'].unique().tolist()
z_able_features = ['Area', 'Mean', 'Mode', 'Min', 'Max', 'Perim.', 'Circ.', 'Feret', 'IntDen', 'Median', 'Round', 'Solidity']

feature_combo_with_zscore=[]
for well in wells:
    this_well = feature_combo.loc[feature_combo['well_x'] == well].copy()
    for feature in z_able_features:
        feature_list = this_well[feature].tolist()
        mean = np.mean(feature_list)
        std = np.std(feature_list)
        this_well[feature + '_well_zscore'] = (this_well[feature] - mean) / std
    feature_combo_with_zscore.append(this_well)

well_zscore_names = [q+'_well_zscore' for q in z_able_features]
oops = ['handle', 'label', 'well_x', 'col_x', 'row_x', 'target_label', 'roi_x', 'lookup']
well_zscore_names = oops + well_zscore_names
area_z = pd.concat(feature_combo_with_zscore)

of_interest = area_z[well_zscore_names]
#%%
# Now I want to look for correlations in my dataset - I think I can use the full seaborn cross plot
  
corr_matrix = of_interest.corr()

fig, ax = plt.subplots(1, 1)
sns.set(rc = {'figure.figsize': (12,12), "figure.dpi":100, 'savefig.dpi':100})
sns.set_style("ticks")

# plotting correlation heatmap 
dataplot = sns.heatmap(of_interest.corr(), cmap="YlGnBu", annot=True, ax=ax)

# Heat map was hard to look at, so getting the high correlations here:
threshold = 0.9
high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                   for i in range(len(corr_matrix)) 
                   for j in range(i+1, len(corr_matrix)) 
                   if abs(corr_matrix.iloc[i, j]) > threshold]

high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])

print(high_corr_df)

# Dropping median and intden as high correlates
#%%
z_cols = of_interest.columns[8:].tolist()
z_cols.remove("IntDen_well_zscore")
z_cols.remove("Median_well_zscore")
#%%
long_list_for_tree = ['lookup', 'target_label','Area', 'Mean', 'Mode', 'Min', 'Max', 'Perim.', 'Circ.', 'Feret', 'Round', 'Solidity'] + z_cols
Use_for_tree = area_z[long_list_for_tree].copy()

# Split the data into features (X) and target (y)
X = Use_for_tree.drop(['target_label', 'lookup'], axis=1)
y = Use_for_tree['target_label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=959)
#%%
param_grid = {
    'n_estimators': [50, 100, 110, 115, 125, 150, 200],
    'max_depth': [None, 10,  15,  20,  25],
    'min_samples_split': [1, 2, 5, 7, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 16],
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

#Use the best from this grid search to do a more local search to find the best!

#%%
import pickle
import os

dump_path = 'C:/Users/ebjam/Documents/GitHub/wormfind/models/'
pickle_path = os.path.join(dump_path, "July_4_random_forest_worm_selection.pkl")
with open(pickle_path, 'wb') as file:
    pickle.dump(best_rf, file)
#%%
import joblib

joblib.dump(best_rf, 'July_4_best_rf_broad_search.joblib')
#%%
# Lets take a crack at linear SVC

use_for_svc = ['Area', 'Mean', 'Mode', 'Min', 'Max', 'Perim.', 'Circ.', 'Feret', 'Round', 'Solidity']
svc_df = area_z[use_for_svc].copy()
targets = area_z['target_label'].copy()
scaler = RobustScaler()

scaled_svc = scaler.fit_transform(svc_df)
scaled_svc_df = pd.DataFrame(scaled_svc, columns = svc_df.columns)

X_train, X_test, y_train, y_test = train_test_split(svc_df, targets, test_size=0.2, random_state=42)


param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'loss': ['hinge', 'squared_hinge'],
    'max_iter': [1000, 2000, 3000, 4000, 10000]
}

linear_svc = LinearSVC()
grid_search_svc = GridSearchCV(estimator=linear_svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV to the scaled training data
grid_search_svc.fit(X_train, y_train)

#%%
from sklearn.linear_model import LogisticRegression

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.0001, 0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 200, 300, 400, 500, 600]
}

grid_searchLR = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_searchLR.fit(X_train, y_train)
#%%
print(f"Best parameters found: {grid_searchLR.best_params_}")

best_LR = grid_searchLR.best_estimator_
y_pred = best_LR.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

#%%

from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'n_neighbors': [3, 4, 5, 6, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
}

grid_searchKN = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_searchKN.fit(X_train, y_train)
#%%
print(f"Best parameters found: {grid_searchKN.best_params_}")

best_KN = grid_searchKN.best_estimator_
y_pred = best_KN.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


#%%

from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_searchGB = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_searchGB.fit(X_train, y_train)

#%%

print(f"Best parameters found: {grid_searchGB.best_params_}")

best_GB = grid_searchGB.best_estimator_
y_pred = best_GB.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


#%%
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_searchXGB = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_searchXGB.fit(X_train, y_train)

#%%

from lightgbm import LGBMClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=LGBMClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

#%%
bad = len(bad_images)
good = len(good_images)
print(bad)
print(good+bad)
print(bad/(good+bad))

