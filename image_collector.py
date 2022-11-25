# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:49:08 2022

@author: ebjam
"""
import os


img_dir = ("C:/Users/ebjam/Documents/GitHub/wormfind/train/")
imglist = os.listdir(img_dir)
img_collection = []
for i in  imglist:
    if i.startswith("72"):
        i_dict={}
        i_dict["license"] = 1
        i_dict["file_name"] = i
        i_dict["height"] = 2208
        i_dict["width"] = 2752
        i_dict["id"] = str(i[0:-4])
        img_collection.append(i_dict)
        
        
#%%
import json

with open(img_dir + "72_train_.json", "w") as outfile:
    json.dump(img_collection, outfile)