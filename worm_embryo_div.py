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
working_dir = "C:/Users/ebjam/Desktop/labeled/train/dapi/" # /leave/a/trailing/slash/
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
#%%The whole thing written horribly with no functions !!!UNTRANSPOSED!!!
#Loop though annotations
for anno in anno_list:
    print(anno)
    im=cv2.imread(working_dir + anno[0:-5] + ".png")
    #open image as 'im' - call this back at the end of the loop!
    #Open annotation as a json
    with open(working_dir + anno) as json_data: #operating within this annotation
        current_json = json.load(json_data)
        #making a labelme format json to write just worm, embryo, and combined crop anno annotations into
        wormlist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        wormlist["imagePath"] = current_json["imagePath"]
        wormlist["imageData"] = current_json["imageData"]
        embryolist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        embryolist["imagePath"] = current_json["imagePath"]
        embryolist["imageData"] = current_json["imageData"]
        worm_count = 0
        #For each shape in json
        for i in range(0, len(current_json["shapes"])):
            #If its a 
            worm_count = 0
            embryo_count = 0
            if current_json["shapes"][i]["label"] == "worm":
                #Addit it to the worm list
                wormlist["shapes"].append(current_json["shapes"][i])
                worm_count +=1

            if current_json["shapes"][i]["label"] == "embryo": # If it's an embryo
                embryolist["shapes"].append(current_json["shapes"][i]) # Add it to the embryo list
                embryo_count +=1
                
            for j in range(0, len(wormlist["shapes"])): #j iterates through worms - select single worm
                croplist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752} #single worm crop
                points = wormlist["shapes"][j]["points"] #points from each worm
                worm_bbox = bounding_box(points) #Bounding box from worms - need these to look for the relevant embryos
                crop_image = im[worm_bbox[1]:worm_bbox[0], worm_bbox[3]:worm_bbox[2]]
                crop_image_name = working_dir + anno[0:-5] + "_worm_" + str(j+1) + ".png"
                cv2.imwrite(working_dir + anno[0:-5] + "_worm_" + str(j+1) + ".png", crop_image)
                croplist["imagePath"] = crop_image_name
                #AFTER bbox, transpose points in worm list

#%% For some reason I don't understand at the moment (probably bad variable naming and not using functions), I need to seperate saving the image and transposing and saving the annotation
for anno in anno_list:
    print(anno)
    with open(working_dir + anno) as json_data: #operating within this annotation
        current_json = json.load(json_data)
        
        wormlist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        wormlist["imagePath"] = current_json["imagePath"]
        wormlist["imageData"] = current_json["imageData"]
        embryolist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
        embryolist["imagePath"] = current_json["imagePath"]
        embryolist["imageData"] = current_json["imageData"]
        worm_count = 0
        #For each shape in json
        for i in range(0, len(current_json["shapes"])):
            #If its a 
            worm_count = 0
            embryo_count = 0
            if current_json["shapes"][i]["label"] == "worm":
                #Addit it to the worm list
                wormlist["shapes"].append(current_json["shapes"][i])
                worm_count +=1

            if current_json["shapes"][i]["label"] == "embryo": # If it's an embryo
                embryolist["shapes"].append(current_json["shapes"][i]) # Add it to the embryo list
                embryo_count +=1
            
            for j in range(0, len(wormlist["shapes"])):   
                croplist = {"version": "4.5.9","flags": {},"shapes":[], "imagePath": "", "imageData": "", "imageHeight": 2208,"imageWidth": 2752}
                points = wormlist["shapes"][j]["points"]
                worm_bbox = bounding_box(points)
                for m in range(0, len(points)):#m iterates over points in the WORM
                    points[m][0] = points[m][0] - worm_bbox[3] #Transposed x = old x - bbox xmin
                    points[m][1] =  points[m][1] - worm_bbox[1] #Transposed y = old y - bbox ymin
                wormlist["shapes"][j]["points"] = points #The old switcheroo to put transposed points into the wormlist
                croplist["shapes"].append(wormlist["shapes"][j])#update croplist - annotation for cropped image - with transposed worm points
                for k in range(0, len(embryolist["shapes"])): #k iterates over embryos
                    embryopoints = embryolist["shapes"][k]["points"] #Get points from embryo
                    emb_bbox = bounding_box(embryopoints) #Get bbox from embryo
                    #print(emb_bbox[0])
                    #print(bbox[0])
                    if emb_bbox[0] <= worm_bbox[0]: #If embryo xmax is smaller than worm xmax
                        if emb_bbox[1] >= worm_bbox[1]:#If embryo xmin is larger than worm xmin
                            if emb_bbox[2] <= worm_bbox[2]:#If embryo ymax is smaller than worm ymax
                                if emb_bbox[3] >= worm_bbox[3]:#If embryo ymin is larger than worm ymin
                                    #Transpose embryo points to cropped worm bounding box 
                                    for l in range(0, len(embryopoints)): #l iterates over points within a shape
                                        embryopoints[l][0] = embryopoints[l][0] - worm_bbox[3] #transposed x = oldx - worm xmin
                                        if embryopoints[l][0] < 0:
                                            print("Ed, you fucked something up transposing the x coord")
                                        embryopoints[l][1] = embryopoints[l][1] - worm_bbox[1] #transposed y = oldy - worm ymin
                                        if embryopoints[l][1] < 0:
                                            print("Ed, you fucked something up transposing the y coord")
                                    croplist["shapes"].append(embryolist["shapes"][k]) #Add untransposed embryo to crop file
                width = worm_bbox[2]-worm_bbox[3] #Image width = xmax - xmin
                height = worm_bbox[0] - worm_bbox[1] #Image height = ymax - ymin
                croplist["imageHeight"] = width
                croplist["imageWidth"] = height
                crop_image_name = working_dir + anno[0:-5] + "_worm_" + str(j+1) + ".png"
                croplist["imagePath"] = crop_image_name
                with open(working_dir + anno[0:-5] + "worm_"+ str(j+1) + "_embryos_transposed.json", "w") as outfile: #End of the "for j..." loop on line 59(or close to) - j iterates over worms
                    json.dump(croplist, outfile)
#%%
            writelist = croplist
            for l in range(0, len(writelist["shapes"])): # number of shapes within worm bbox -  one worm, many embryos
                if writelist["shapes"][l]["label"] == "worm": #if the shape is a worm
                    wormbbox = bounding_box(croplist["shapes"][l]["points"]) #Get the bounding box from that worm
                    width = wormbbox[2]-wormbbox[3] #Image width = xmax - xmin
                    height = wormbbox[0] - wormbbox[1] #Image height = ymax - ymin
                    writelist["imageHeight"] = width
                    writelist["imageWidth"] = height
                global_points = croplist["shapes"][l]["points"] #For any shape - get the points - global - i.e. within uncropped image
                localpoints = [] #localpointS object
                        #print(global_points[0])
                for m in range(0, len(global_points)): #For length of points (single point is x,y)
                    global_point = global_points[m]
                    #print(global_points)
                    #print(global_point)
                    localpoint = [0 , 0] #localpoint object - list of two items
                    localpoint[0] = global_points[m][0] #Local point x = global 'points' m x coord - is global address
                    #print(localpoint[0])
                    localpoint[0] = localpoint[0] - wormbbox[3] #Local x coord = global xcoord - wormbbox xmin
                    #print(localpoint[0])
                    localpoint[1] = global_points[m][1]
                    localpoint[1] = localpoint[1] - wormbbox[1]
                    localpoints.append(localpoint)
                        
                writelist["shapes"][l]["points"] = localpoints
                with open(working_dir + anno[0:-5] + "worm_"+ str(i) + "_embryos.json", "w") as outfile:
                    json.dump(writelist, outfile)
        for i in range(0, len(wormlist["shapes"])):
            points = wormlist["shapes"][i]["points"]
            bbox = bounding_box(points)
            #Crop input image to bounding box
            crop_image = im[bbox[1]:bbox[0], bbox[3]:bbox[2]]
            cv2.imshow("", crop_image)
            cv2.waitKey(0)
            cv2.imwrite(working_dir + anno[0:-5] + "_worm_" + str(i) + ".png", crop_image)
#%%
writelist = croplist

#%%
for l in range(0, len(writelist["shapes"])): # number of shapes within worm bbox -  one worm, many embryos
    if writelist["shapes"][l]["label"] == "worm": #if the shape is a worm
        wormbbox = bounding_box(croplist["shapes"][l]["points"]) #Get the bounding box from that worm
        #print(wormbbox)
        width = wormbbox[2]-wormbbox[3] #Image width = xmax - xmin
        height = wormbbox[0] - wormbbox[1] #Image height = ymax - ymin
        writelist["imageHeight"] = width
        writelist["imageWidth"] = height
        #print(width, height)
    global_points = croplist["shapes"][l]["points"] #For any shape - get the points - global - i.e. within uncropped image
    localpoints = [] #localpointS object
    #print(global_points[0])
    for m in range(0, len(global_points)): #For length of points (single point is x,y)
        global_point = global_points[m]
        #print(global_points)
        #print(global_point)
        localpoint = [0 , 0] #localpoint object - list of two items
        localpoint[0] = global_points[m][0] #Local point x = global 'points' m x coord - is global address
        #print(localpoint[0])
        localpoint[0] = localpoint[0] - wormbbox[3] #Local x coord = global xcoord - wormbbox xmin
        #print(localpoint[0])
        localpoint[1] = global_points[m][1]
        localpoint[1] = localpoint[1] - wormbbox[1]
        localpoints.append(localpoint)
        
    writelist["shapes"][l]["points"] = localpoints
    with open(working_dir + anno[0:-5] + "worm_"+ str(i) + "_embryos.json", "w") as outfile:
        json.dump(writelist, outfile)
#%%
                for a in range(0, len(croplist['shapes'])): # Minus xmin and ymin from each xy point in embryo to transpose annotation to cropped image - 'a' is # of all all shapes
                    print(croplist['shapes'][a]["points"])
                    for b in range(0, len(croplist['shapes'][a]["points"])):#Get number of points within each shape - each point is a list of xy
                        print("before: " + str(croplist['shapes'][a]["points"][b][0]))
                        print(bbox[0])
                        croplist['shapes'][a]["points"][b][0] = croplist['shapes'][a]["points"][b][0] - bbox[0] #(pointx) = pointx - box x
                        print("after: " + str(croplist['shapes'][a]["points"][b][0]))
                        croplist['shapes'][a]["points"][b][1] = croplist['shapes'][a]["points"][b][1] - bbox[2] #(pointy) = pointy - box y
                        #Should produce a crop list - complete annotations without height and width 
                        #Pull height and width from worm bbox
                        h = bbox[1] - bbox[0]
                        print("height = ", h)
                        w = bbox[2] - bbox[3]
                        croplist["imageHeight"] = h
                        croplist["imageWidth"] = w
                #Dump croplist!
                #with open(working_dir + anno[0:-5] + "worm_"+ str(i) + "_embryos.json", "w") as outfile:
                    #json.dump(croplist, outfile)
#%%
        for i in range(0, len(wormlist["shapes"])):
            points = wormlist["shapes"][i]["points"]
            bbox = bounding_box(points)
            #Crop input image to bounding box
            crop_image = im[bbox[1]:bbox[0], bbox[3]:bbox[2]]
            cv2.imshow("", crop_image)
            cv2.waitKey(0)
            cv2.imwrite(working_dir + anno[0:-5] + "_worm_" + str(i) + ".png", crop_image)