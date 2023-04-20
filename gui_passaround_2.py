# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:21:53 2023

@author: ebjam
"""

import pickle
import os
import numpy as np
#import cv2

# Import the package if saved in a different .py file else paste 

start_folder = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/combo/input/png/test+val/full/DAPI/"

# Probably don't need to change full_image_pickle!
full_image_pickle = "full_image_results2.pickle"
index_segmentation_record = os.path.join(start_folder, full_image_pickle)
file = open(index_segmentation_record,'rb')
seg_record = pickle.load(file)


def transpose_segmentation(bbox, segmentation):
    minx = bbox[0]
    miny = bbox[1]
    for i in segmentation:
        i[0] = i[0] - minx
        i[1] = i[1] - miny
    #Segmentation is now transposed to bbox
    return(segmentation)


for q in range(0, len(seg_record['image_titles'])-1):
    image = seg_record['image_titles'][q]
    result = seg_record['results'][q]
    single_worms = []
    masks = result.masks.xy
    boxes = result.boxes.xyxy
    if len(masks) < 1:
        print('No worms detected in image ' + image)
    else:
        for w in range(0, len(masks)-1):
            singleworm = {}
            maskcopy = masks[w].copy()
            singleworm['segmentation'] = maskcopy
            singleworm['bbox'] = boxes[w].tolist()
            singleworm['transposed_segmentation'] = transpose_segmentation(boxes[w], masks[w])
            singleworm['title'] = image
            singleworm['wormID'] = image[:-4] + 'worm_' + str(w)
            single_worms.append(singleworm)
    image_record = {}
    image_record['input_image'] = image
    image_record['single_worms'] = single_worms
    
    with open(start_folder + "DY96_labmeeting/" + image[:-4] + "NO_GUI.result", "wb") as f:
        pickle.dump(image_record, f, protocol=pickle.HIGHEST_PROTOCOL)

#%%
# Probably don't need to change full_image_pickle!
full_image_pickle = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/combo/input/png/test+val/full/DAPI/DY96_labmeeting/72hrn2i_50u-08.result"
index_segmentation_record = os.path.join(start_folder, full_image_pickle)
file = open(index_segmentation_record,'rb')
single_record = pickle.load(file)



