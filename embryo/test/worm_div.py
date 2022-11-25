# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:41:27 2022

@author: ebjam
"""
#Imports
import json
import os
import math
import cv2
#%% Set up dir and lists
working_dir = "C:/Users/ebjam/Desktop/labeled/test/"
file_list = os.listdir(working_dir)
img_list = [] 
anno_list = [] 

#%%Divide file list into images and jsons
for file in file_list:
    if file[-1] == 'g':
        img_list.append(file)
    elif file[-1] =='n':
        anno_list.append(file)

#%%Bounding box function lightly adapted from https://stackoverflow.com/questions/46335488/how-to-efficiently-find-the-bounding-box-of-a-collection-of-points
def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return [math.ceil(max(y_coordinates)), math.floor(min(y_coordinates)), math.ceil(max(x_coordinates)), math.floor(min(x_coordinates))]
#%%
#Loop though annotations
for anno in anno_list:
    im=cv2.imread(working_dir + anno[0:-4] + "png")
    #Open annotation as a json
    with open(working_dir + anno) as json_data:
        current_json = json.load(json_data)
        #making a labelme format json to write just worm annotations into
        wormlist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        wormlist["imagePath"] = current_json["imagePath"]
        wormlist["imageData"] = current_json["imageData"]
        for i in range(0, len(current_json["shapes"])):
            if current_json["shapes"][i]["label"] == "worm":
                wormlist["shapes"].append(current_json["shapes"][i])
        #Saving those worm annotations as their own files
        #with open(working_dir + anno[0:-5] + "_worms.json", "w") as outfile:
            #json.dump(wormlist, outfile)
        #Get bounding box from each shape
        for i in range(0, len(wormlist["shapes"])):
            points = wormlist["shapes"][i]["points"]
            bbox = bounding_box(points)
            #Crop input image to bounding box
            crop_image = im[bbox[1]:bbox[0], bbox[3]:bbox[2]]
            cv2.imwrite(working_dir + anno[0:-5] + "_worm_" + str(i) + ".png", crop_image)