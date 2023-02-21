# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:37:01 2023

@author: ebjam

GUI pass around
"""
import os
import pickle
import numpy as np

def get_bounding_box(segmentation):
    min_x = min(segmentation, key=lambda p: p[0])[0]
    max_x = max(segmentation, key=lambda p: p[0])[0]
    min_y = min(segmentation, key=lambda p: p[1])[1]
    max_y = max(segmentation, key=lambda p: p[1])[1]
    return [min_x, min_y, max_x, max_y]

def transpose_segmentation(bbox, segmentation):
    minx = bbox[0]
    miny = bbox[1]
    for i in segmentation:
        '''
        i[0] * w turns decimal location into real location in input image. 
        i.e. coordinate of x = 0.567 on an image 1000 pixels wide would be pixel 567
        
        '- minx' and '-miny' transposes segmentation to the bbox rather than the full image
        '''
        i[0] = i[0] - minx
        i[1] = i[1] - miny
    #Segmentation is now transposed to bbox
    return(segmentation)

start_folder = "C:/Users/ebjam/Downloads/gui testers-20230213T211340Z-001/gui testers/"
full_image_pickle = "full_image_results.pickle"
index_segmentation_record = os.path.join(start_folder, full_image_pickle)
file = open(index_segmentation_record,'rb')
seg_record = pickle.load(file)

for q in range(0, len(seg_record["image_titles"])):
    #Open DY96 image here!
    im_title = seg_record["image_titles"][q]
    annotations = {"single_worms": [], "input_image": ""}
    annotations["input_image"] = im_title
    results = seg_record["results"][q]
    masks = results.masks.segments
    annotation = {}
    #Initiate worm counting
    worm_no = 0
    #For worm in image
    for seg in masks:
        points = []
        #For point in single worm - un-normalize and bring back to image
        for point in seg:
            point=point.tolist()
            point[0] = point[0] * 2752 #point[0] * w
            points.append(point[0])
            point[1] = point[1] * 2208 #point[1] * h
            points.append(point[1])
        #Things come out of the GUI as a nparray - doing that with this one line below
        save_points = np.array([[points[i], points[i+1]] for i in range(0, len(points), 2)])
        #Same save title as GUI - will be name for DY96 crop
        save_title = im_title[:-4] +"worm_" + str(worm_no)
        annotation["title"] = im_title
        annotation["wormID"] = save_title
        annotation["bbox"] = get_bounding_box(save_points)
        annotation["segmentation"] = save_points
        annotation["transposed_segmentation"] = transpose_segmentation(annotation["bbox"], annotation["segmentation"])
        annotations["single_worms"].append(annotation)
        # Crop and save DY96 image - will pass to microsporidia and embryo detection
        worm_no +=1