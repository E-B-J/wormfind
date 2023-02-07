# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:43:13 2023

@author: ebjam
"""
from ultralytics import YOLO
import os
import cv2
import csv
from datetime import date
import json
from shapely.geometry import Point, Polygon

today = date.today()

model_string = "/path/to/model/"
input_folder = "path/to/folder/full/of/cropped/DY96worms" 

def load_model(model_string):
    model = YOLO(model_string)
    return(model)

def list_out_images(input_folder):
    wormsegs = input_folder + "worm_segmentations.json"
    worm_segmentations = json.load(wormsegs)
    image_list = [q for q in os.listdir(input_folder) if q.endswith(".png")]
    image_array_list = []
    for image in image_list:
        img_array = cv2.imread(input_folder + image)
        image_array_list.append(img_array)
    id_and_array = dict(zip(image_list, image_array_list))
    return(id_and_array, worm_segmentations)

def find_centers(theboxes):
    centerpoints = []
    for box in theboxes:
        x1, y1, x2, y2 = box.xyxy
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2
        cp = Point(cx, cy)
        centerpoints.append(cp)
    return(centerpoints)

id_and_array = list_out_images(input_folder)

def predict_embryos(id_and_array, worm_segmentations):
    worm_seg_and_egg = worm_segmentations
    model = load_model(model_string)
    for key in id_and_array:
        results = model.predict(source = id_and_array[key], save=False, save_txt=False)
        worm_seg_and_egg[key]["embryo_bboxes"] = results[0].boxes
        center_points = find_centers(results[0].boxes)
        worm_seg_and_egg[key]["embryo_centers"] = center_points
    return(worm_segmentations)

def check_gravidity(worm_seg_and_egg):
    for e in range(0, len(worm_seg_and_egg)):
        seg = worm_seg_and_egg[e]['segmentation'] #!!!
        embryo_centerpoints = worm_seg_and_egg[e]["embryo_centers"]
        polygon = Polygon(seg) #!!!
        for center_point in embryo_centerpoints:
            if polygon.contains(center_point):
                if "embryo_no" not in worm_seg_and_egg[e]:
                    # Make keypoint list
                    worm_seg_and_egg[e]["embryo_no"] = 0
                    worm_seg_and_egg[e]["gravidity"] = 0
                    worm_seg_and_egg[e]["internal_embryos"] = []
                    # Add segmentation to keypoint list, have to add 'v' coord (x, y, v):
                    # 0 = not labelled, 1 = labeled not visable, 2 = labelled and visible
                worm_seg_and_egg[e]['embryo_no'] += 1
                worm_seg_and_egg[e]["internal_embryos"].append(center_point)
        if worm_seg_and_egg[e]["embryo_no"] > 0:
            worm_seg_and_egg[e]["gravidity"] = 1
    return(worm_seg_and_egg)

#%%
def detector(inputfolder, embryos = 0, microsporidia = 0):
    id_and_array, worm_segments = list_out_images(input_folder)
    if embryos == 1:
        embryo_predicted_json_guy = check_gravidity(predict_embryos(id_and_array, worm_segments))    
    if microsporidia == 1:
        print("Ed still has to write this bit!")
    if embryos == 1:
        if microsporidia == 1:
            #line writer for all!
            print('lw1')
        elif microsporidia == 0:
            #Linewriter for embryos only
            print('lw2')
    elif embryos == 0 and microsporidia == 1:
        print('lw3')
        #Line writer for microsporidia alone
#%%

res = detector("input/folder/with/worm/predictions/and/cropped/dy96/", embryos = 1)

