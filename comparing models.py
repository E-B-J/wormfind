# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:35:19 2023

@author: ebjam
"""

import seaborn as sns
import pandas as pd
import scipy.stats as stats
import numpy as np

sns.set(rc = {'figure.figsize': (12,10), "figure.dpi":1000, 'savefig.dpi':1000})
sns.set_style("ticks")

#%%

data = pd.read_csv("C:/Users/ebjam/Documents/Toronto/automation attempts/100trees_100branches/overlapImagefull_fat_and_intensity.csv")
data_2 = pd.read_csv("C:/Users/ebjam/Documents/GitHub/wormfind/rf/featureimportance_20220116.csv")

#%%
full_fat = data.loc[data["model"] == "full fat"]
skim = data.loc[data["model"] == "skim"]

#%%
data = skim
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)

mean_pre = [data["Overlap_Precision_pred_threshold"].mean()]
axes[0].axhline(y = mean_pre, color = 'lightgrey', zorder = 0, ls = '--')
mean_rec = data["Overlap_Recall_pred_threshold"].mean()
axes[1].axhline(y = mean_rec, color = 'lightgrey', zorder = 0, ls = '--')

sns.violinplot(y = "Overlap_Precision_pred_threshold", x="utility", data = data, ax = axes[0], cut = 0, inner=None, zorder = 0, color = "forestgreen")#"lightblue")
sns.stripplot(y = "Overlap_Precision_pred_threshold", data = data, ax = axes[0], color = 'slategrey', zorder = 1)
mean_pre = [data["Overlap_Precision_pred_threshold"].mean()]

sns.pointplot(y = "Overlap_Precision_pred_threshold", data = data, ax = axes[0], errorbar = "sd", capsize= 0.05, color='k', zorder = 3, markers = 'D')


sns.violinplot(y = "Overlap_Recall_pred_threshold", data = data, ax = axes[1], cut = 0, color = "orange", inner = None, zorder = 0)
sns.stripplot(y = "Overlap_Recall_pred_threshold", data = data, ax = axes[1], color = 'slategrey', zorder = 1)
sns.pointplot(y = data["Overlap_Recall_pred_threshold"], ax = axes[1], errorbar = "sd", color = "k", capsize = 0.05, zorder = 3, markers = 'D')


axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[0].set_ylabel("Precision")
axes[1].set_ylabel("Recall")


sns.despine()


#%%

#data = skim
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)

mean_pre = [data["Overlap_Precision_pred_threshold"].mean()]
axes[0].axhline(y = mean_pre, color = 'lightgrey', zorder = 0, ls = '--')
mean_rec = data["Overlap_Recall_pred_threshold"].mean()
axes[1].axhline(y = mean_rec, color = 'lightgrey', zorder = 0, ls = '--')

sns.violinplot(y = "Overlap_Precision_pred_threshold", x = "utility", data = data, ax = axes[0], cut = 0, hue = "model", dodge=True, split = True, inner=None, zorder = 0)#"lightblue")
sns.stripplot(y = "Overlap_Precision_pred_threshold", x = "utility", data = data, dodge=True, hue="model", ax = axes[0], color = 'slategrey', zorder = 1)
mean_pre = [data["Overlap_Precision_pred_threshold"].mean()]

sns.pointplot(y = "Overlap_Precision_pred_threshold", x = "utility", hue = "model", dodge = True, data = data, ax = axes[0], errorbar = "sd", capsize= 0.05, color='k', zorder = 3, markers = 'D')


sns.violinplot(y = "Overlap_Recall_pred_threshold", x = "utility", data = data, ax = axes[1], cut = 0, hue = "model", split = True, inner = None, zorder = 0)
sns.stripplot(y = "Overlap_Recall_pred_threshold", x = "utility", data = data, dodge = True, hue = "model",  ax = axes[1], color = 'slategrey', zorder = 1)
sns.pointplot(y = data["Overlap_Recall_pred_threshold"], x = data["utility"], hue = data["model"], dodge = True, ax = axes[1], errorbar = "sd", color = "k", capsize = 0.05, zorder = 3, markers = 'D')


axes[0].set_xlabel("")
axes[1].set_xlabel("")
axes[0].set_ylabel("Precision")
axes[1].set_ylabel("Recall")


sns.despine()

#%%
from scipy.stats import wilcoxon

stat, p = wilcoxon(full_fat["Overlap_Precision_pred_threshold"], skim["Overlap_Precision_pred_threshold"])
print('Precision - Statistics=%.3f, p=%.3f' % (stat, p))

stat, p = wilcoxon(full_fat["Overlap_Recall_pred_threshold"], skim["Overlap_Recall_pred_threshold"])
print('Recall - Statistics=%.3f, p=%.3f' % (stat, p))
#%%
sns.violinplot(y = "Overlap_Precision_pred_threshold",x = "utility", data = data, cut = 0, hue = data["model"], split = True, inner=None, zorder = 0)
#%%
intensity = data_2.loc[data_2["Category"] == "Intensity"]
eigen1 = data_2.loc[data_2["Category"] == "Eigen Value 1"]
eigen2 = data_2.loc[data_2["Category"] == "Eigen Value 2"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

sns.swarmplot(ax=axes[0], y = intensity["Feature_Importance"], x = intensity["Sigma"], hue = intensity["Model"]) 
sns.swarmplot(ax=axes[1], y = eigen1["Feature_Importance"], x = eigen1["Sigma"], hue = eigen1["Model"], marker = 's') 
sns.swarmplot(ax=axes[1], y = eigen2["Feature_Importance"], x = eigen2["Sigma"], hue = eigen2["Model"], marker = 'o', edgecolor='white', linewidth=1) 

axes[0].axhline(y=1/15, ls="--", color = "lightgrey")
axes[1].axhline(y=1/15, ls="--", color = "lightgrey")
axes[0].get_legend().remove()
axes[1].get_legend().remove()