# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:24:42 2023

@author: ebjam
"""
import os
import pickle
from shapely.geometry import Polygon
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#%%
segmentation_record = "C:/Users/ebjam/Downloads/gui_testers-20230213T211340Z-001/test_size_filter/full_image_results.pickle"
file = open(segmentation_record,'rb')
seg_record = pickle.load(file)
'''
all_masks = []
for result in seg_record["results"]:
    this = result
    masks = result.masks.segments
    for mask in masks:
        all_masks.append(mask)
     
areas = []
for mask in all_masks:
    #unnormalize
    for i in mask:
        i[0] = (i[0] * 2752 * 1.1013)
        i[1] = (i[1] * 2208 * 1.1013)
    mask = [tuple(x) for x in mask]
    if mask[0] != mask[len(mask)-1]:
        mask.append(mask[0])
    areas.append(Polygon(mask).area)

'''
#%%
'''
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
sns.violinplot(areas, ax = axes[0], cut = 0, inner = None, zorder = 0)
sns.stripplot(areas, ax = axes[0], color = 'slategrey', edgecolor='k', linewidth = 1, zorder = 1)
area_mean = (sum(areas)/len(areas))
filtered_areas = [q for q in areas if q > 10000]
sns.violinplot(filtered_areas, ax = axes[1], cut = 0, inner = None, color = "forestgreen", zorder = 0)
sns.stripplot(filtered_areas, ax = axes[1], color = 'slategrey', edgecolor='k', linewidth = 1, zorder = 1)
sns.pointplot(x = areas, ax = axes[0], color = 'red', errorbar = None, ci = None, markersize = 10, edgecolor='k', linewidth = 0, zorder = 10)
sns.pointplot(x = filtered_areas, ax = axes[1], color = 'red', errorbar = None, ci = None, markersize = 10, edgecolor='k', linewidth = 0, zorder = 10)
axes[0].set_title("Unfiltered, N = " + str(len(areas)))
axes[0].set_xlabel("Worm area um^2")
axes[1].set_xlabel("Worm area um^2")
axes[1].set_title("Filtered at 10000um^2,  N = " + str(len(filtered_areas)))
axes[0].set_ylabel("Density", fontsize = 20)
sns.despine()
'''
#%%
'''
def size_filter(full_image_results):
    for result in full_image_results['results']:
        print(len(result.masks.segments))
        small_shape_index = []
        for q in range(0, len(result.masks.segments)):
            mask = result.masks.segments[q]
            tempmask = mask.copy()
            #print(tempmask)
            for i in tempmask:
                #Unnormalize to pixels, and scale to microns
                i[0] = (i[0] * 2752 * 1.1013)
                i[1] = (i[1] * 2208 * 1.1013)
                        #Tuple for polygon
            mask_for_poly = [tuple(x) for x in tempmask]
            #Close polygon
            if mask_for_poly[0] != mask_for_poly[len(mask_for_poly)-1]:
                mask_for_poly.append(mask_for_poly[0])
            area = (Polygon(mask_for_poly).area)
            filter_size = 10000
            
            if area <= filter_size:
                #Changes length of masks!! Need to do this to a temp holder!
                small_shape_index.append(q)
        #SSI contains indeces to remove
        #Make SSI go from high to low, then pop in order
        small_shape_index.sort(reverse=True)
        for w in range(0, len(small_shape_index)):
            result.masks.segments.pop(small_shape_index[w])
        print(len(result.masks.segments))   
    return(full_image_results)
'''

#%%
'''
filter_dict = size_filter(seg_record)
'''