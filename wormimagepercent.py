# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:33:10 2022

@author: ebjam
"""
import json
#%%
def group_to_indie(workingjson):
    with open(workingjson) as test_json:
        test_set = json.load(test_json)
        sep_dict = {}
        for r in range(0, len(test_set["images"])):
        #extract image ID and name into a dictionary to be sure of ID later
            key = test_set["images"][r]['id']
            #imlist[test_set["images"][r]['file_name']] = test_set["images"][r]['id']
            temp_anno = {"annotations": []}
            sep_dict[test_set["images"][r]['file_name']] = {}
            for s in range(0, len(test_set["annotations"])):
                if test_set["annotations"][s]["image_id"] == key:
                    print("Sorted annotation", s, "to image " + key)
                    temp_anno["annotations"].append(test_set["annotations"][s])
            sep_dict[test_set["images"][r]['file_name']] = temp_anno
            sep_dict[test_set["images"][r]['file_name']]["categories"] = test_set["categories"]
            sep_dict[test_set["images"][r]['file_name']]["images"] = test_set["images"][r]
    return(sep_dict)

#%%
trainset = group_to_indie("C:/Users/ebjam/Documents/GitHub/wormfind/train_json_4872.json")
#%%
total_area = 2208*2752
area_list = []
worms = 0
for q in trainset: # i is an image
    img_worm_area = 0
    for annotation in trainset[q]["annotations"]:
        img_worm_area += annotation["area"]
        worms +=1
    area_list.append(100*(img_worm_area/total_area))
    
#%%
def Average(lst):
    return sum(lst) / len(lst)

mean = Average(area_list)
#%%
import seaborn as sns
import statistics

sns.set(rc = {'figure.figsize': (16,9), "figure.dpi":1000, 'savefig.dpi':1000})
sns.set_style("ticks")
ax = sns.distplot(area_list)
ax.set_xlim(0, 30)
ax.set_xlabel("Percent of image taken up by C. elegans")
ax.axvline(statistics.median(area_list), color = "r")
sns.despine()
#%%
print(statistics.median(area_list))