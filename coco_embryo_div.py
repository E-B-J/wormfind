# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:41:27 2022

@author: ebjam
"""
#Imports
import json
import math
import cv2

#%% Set up dir and lists
working_dir = "C:/Users/ebjam/Desktop/labeled/test/" # /leave/a/trailing/slash/
#%%Breaking down coco to individual images
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
                    temp_anno["annotations"].append(test_set["annotations"][s])
            sep_dict[test_set["images"][r]['file_name']] = temp_anno
            sep_dict[test_set["images"][r]['file_name']]["categories"] = test_set["categories"]
            sep_dict[test_set["images"][r]['file_name']]["images"] = test_set["images"][r]
    return(sep_dict)
#%%
modtest = group_to_indie("C:/Users/ebjam/Desktop/labeled/test.json")  
#%%
working_dir = "C:/Users/ebjam/Desktop/labeled/test/"
transposed_embryo_out = {"annotations": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}], "images": []}
wormcount = 0

for q in modtest:
    selected_image = modtest[q]
    imagewormlist = []
    imageemblist = []
    for w in range(0, len(selected_image["annotations"])):
        #for all annos 
        if selected_image["annotations"][w]["category_id"] == 2: #2 is worm, 1 is embryo, 0 is cluster
            imagewormlist.append(selected_image["annotations"][w])
        elif selected_image["annotations"][w]["category_id"] == 1:  
            imageemblist.append(selected_image["annotations"][w])
    for e in range(0, len(imagewormlist)):
        imagecount = 0
        worm_img_annotations = {"annotations": []}
        single_worm_bbox = imagewormlist[e]["bbox"]
        im = cv2.imread(working_dir + q)
        #im = cv2.flip(im, 0)
        crop_image = im[math.floor(single_worm_bbox[1]):math.ceil(single_worm_bbox[1] + single_worm_bbox[3]), math.floor(single_worm_bbox[0]):math.ceil(single_worm_bbox[0]+single_worm_bbox[2])]
        cv2.imwrite(working_dir + q[0:-4] + "_worm_" + str(e+1) + ".png", crop_image)
        keyname = q[0:-4] + "_worm_" + str(e+1) + ".png"
        imdict = {}
        imdict["height"] = single_worm_bbox[3]
        imdict["width"] = single_worm_bbox[2]
        imdict["id"] = wormcount
        imdict["file_name"] = keyname
        transposed_embryo_out["images"].append(imdict)
        for r in range(0, len(imageemblist)):
            embryo_bbox = imageemblist[r]["bbox"]
            if embryo_bbox[0] > single_worm_bbox[0]:
                if (embryo_bbox[0] + embryo_bbox[2]) < (single_worm_bbox[0] + single_worm_bbox[2]):
                    if embryo_bbox[1] > single_worm_bbox[1]:
                        if (embryo_bbox[1] + embryo_bbox[3]) < (single_worm_bbox[1] + single_worm_bbox[3]):
                            embryo_bbox[0] = embryo_bbox[0] - single_worm_bbox[0]
                            embryo_bbox[1] = embryo_bbox[1] - single_worm_bbox[1]
                            imageemblist[r]["bbox"] = embryo_bbox
                            imageemblist[r]["image_id"] = wormcount
                            for y in range(0, len(imageemblist[r]["segmentation"][0])):
                                    imageemblist[r]["segmentation"][0][y] = imageemblist[r]["segmentation"][0][y] - single_worm_bbox[y%2]

        transposed_embryo_out["annotations"].append(imageemblist[r])
        wormcount += 1
        
with open(working_dir + "worm_by_worm_embryo_test_transposed.json", "w") as outfile: #End of the "for j..." loop on line 59(or close to) - j iterates over worms
    json.dump(transposed_embryo_out, outfile)