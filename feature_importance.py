# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:23:37 2023

@author: ebjam
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines #Visualization util
import matplotlib.patches as mpatches

#%%Load data
data = pd.read_csv("C:/Users/ebjam/Documents/GitHub/wormfind/rf/featureimportance_20220116.csv")

intensity = data.loc[data["Category"] == "Intensity"]
strucure1 = data.loc[data["Category"] == "Eigen Value 1"]
structure2 = data.loc[data["Category"] == "Eigen Value 2"]

#%%

modellist = intensity.Model.unique().tolist()
struc2_models = []
for model in modellist:
    mod_data = structure2.loc[structure2["Model"] == model]
    struc2_models.append(mod_data)
    

#%%
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
length = len(intense_models)

for ch, color in zip(range(length), ["#ceff1a","#aaa95a","#82816d","#414066","#1b2d2a"]):
    current_i_data = intense_models[ch]
    axes[0].plot(current_i_data["Sigma"], current_i_data["Feature_Importance"], 'o', color=color)
    axes[0].set_title("Intensity features")
    axes[0].set_xlabel("$\\sigma$")
    current_s1_data = struc1_models[ch]
    axes[1].plot(current_s1_data["Sigma"], current_s1_data["Feature_Importance"], 's', color=color)
    current_s2_data = struc2_models[ch]
    axes[1].plot(current_s2_data["Sigma"], current_s2_data["Feature_Importance"], 'P', color=color)
    axes[1].set_title("Structure features")
    axes[1].set_xlabel("$\\sigma$")

mean_importance = 1/15
axes[0].axhline(y=mean_importance, color='darkgrey', linestyle='--')
axes[1].axhline(y=mean_importance, color='darkgrey', linestyle='--')
axes[0].set_xticks(np.arange(0, 20, 2))
axes[1].set_xticks(np.arange(0, 20, 2))
axes[0].set_ylabel("Feature Importance")

sns.despine()
model_label_list = ["100 trees, 10 branches", "200 trees, 12 branches", "200 trees, 14 branches", "60 trees, 10 branches", "200 trees, 10 branches"]
patches = []
for color, model in zip(["#ceff1a","#aaa95a","#82816d","#414066","#1b2d2a"], model_label_list):
    patch = mpatches.Patch(color = color, label = model)
    patches.append(patch)
meanimportance = mlines.Line2D([], [], color='darkgrey', marker = 'D', linestyle='--',
                          markersize=0, label='Mean Feature importance')
patches.append(meanimportance)
fig.legend(loc = "upper center",handles=patches, bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True)