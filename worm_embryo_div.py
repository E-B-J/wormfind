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
working_dir = "C:/Users/ebjam/Desktop/labeled/train/"
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
    im=cv2.imread(working_dir + anno[0:-5] + "DY96.png") #open image as 'im' - call this back at the end of the loop!
    #Open annotation as a json
    with open(working_dir + anno) as json_data:
        current_json = json.load(json_data)
        #making a labelme format json to write just worm, embryo, and combined crop anno annotations into
        wormlist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        wormlist["imagePath"] = current_json["imagePath"]
        wormlist["imageData"] = current_json["imageData"]
        embryolist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        embryolist["imagePath"] = current_json["imagePath"]
        embryolist["imageData"] = current_json["imageData"]
        croplist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        #For each shape in json
        for i in range(0, len(current_json["shapes"])):
            #If its a worm
            if current_json["shapes"][i]["label"] == "worm":
                #Addit it to the worm list
                wormlist["shapes"].append(current_json["shapes"][i])
            if current_json["shapes"][i]["label"] == "embryo": # If it's an embryo
                embryolist["shapes"].append(current_json["shapes"][i]) # Add it to the embryo list
            for i in range(0, len(wormlist["shapes"])): #loop through worms
                croplist["shapes"].append = wormlist["shapes"][i] #Add worm to croplist (Just worm shapes and v/f tags)
                points = wormlist["shapes"][i]["points"] #points from each worm
                bbox = bounding_box(points) #Bounding box from worms - need these to look for teh relevant embryos
                for i in range(0, len(embryolist["shapes"])): #iterating over all embryos
                    embryopoints = embryolist["shapes"][i]["points"] #Get points from embryo
                    emb_bbox = bounding_box[embryopoints] #Get bbox from embryo
                    if emb_bbox[0] >= bbox[0]: #If embryo xmax is smaller than worm xmax
                        if emb_bbox[1] <= bbox[1]:#If embryo xmin is larger than worm xin
                            if emb_bbox[2] >= bbox[2]:#If embryo ymax is smaller than worm ymax
                                if emb_bbox[3] <= bbox[3]:#If embryo ymin is larger than worm ymin
                                    croplist.append(embryolist["shapes"][i]["points"]) #Add untransposed embryo to crop file
                for a in croplist: # Minus xmin and ymin from each xy point in embryo to transpose annotation to cropped image
                    for b in range(0, len(croplist['shapes'])): #Get number of embryos within the crop + the worm added first
                        croplist['shapes'][b]["points"][0] = croplist['shapes'][b]["points"][0] - bbox[0]
                        croplist['shapes'][b]["points"][1] = croplist['shapes'][b]["points"][1] - bbox[2]
                        #Should produce a crop list - complete annotations without height and width 
                        #Pull height and width from worm bbox
                        h = bbox[1] - bbox[0]
                        w = bbox[2] - bbox[3]
                        croplist["imageHeight"] = h
                        croplist["imageWidth"] = w
                #Dump croplist!
                with open(working_dir + anno[0:-5] + "one_worm_many_embryos.json", "w") as outfile:
                    json.dump(croplist, outfile)
        for i in range(0, len(wormlist["shapes"])):
            points = wormlist["shapes"][i]["points"]
            bbox = bounding_box(points)
            #Crop input image to bounding box
            crop_image = im[bbox[1]:bbox[0], bbox[3]:bbox[2]]
            cv2.imwrite(working_dir + anno[0:-5] + "DY96_worm_" + str(i) + ".png", crop_image)