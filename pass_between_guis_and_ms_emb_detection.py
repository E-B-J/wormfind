# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:10:07 2023

@author: ebjam
"""

import os
import cv2
import pickle

dy96path = "path/to/dy96/"

result_list = [q for q in os.listdir(dy96path) if q.endswith(".result")]
for result_file in result_list:
    #Load result file
    segmentation_record = os.path.join(dy96path, result_file)
    file = open(segmentation_record,'rb')
    seg_record = pickle.load(file)
    
    #Load image with cv2
    dy96_image = cv2.imread(os.joinpath(dy96path, seg_record["input_image"]))
    #Make list of annotations within image
    annotations_in_image = seg_record["single_worms"]
    #Pull save title and bbox from each worm - bbox to crop, save title to save crop if needed
    for w in range(0, len(annotations_in_image)):
        save_title = annotations_in_image[w]["save_title"]
        bbox = annotations_in_image[w]["bbox"]
        #Crop each worm's bbox, and send it to detection!
        crop_dy96 = dy96_image[bbox[2]:bbox[3], bbox[0]:bbox[1]]
        #Could save crop here with:
        #cv2.imwrite(os.joinpath(dy96path, save_title), crop_dy96)
        #Send crop_dy96 to microsporidia and embryo detection!!
        