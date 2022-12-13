# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:41:27 2022
completed... nov 30
@author: ebjam
"""
#Imports
import json
import math
from pycocotools.coco import COCO
import cv2

#%%
def bbox_relation(wormbbox, embryobbox):
    if wormbbox[0] <= embryobbox[0] and wormbbox[1] <= embryobbox[1]:
        if (wormbbox[0] + wormbbox[2]) > (embryobbox[0] + embryobbox[2]):
            if (wormbbox[1] + wormbbox[3]) > (embryobbox[1] + embryobbox[3]):
                inside = True
            else:
                inside = False
        else:
            inside = False
    else:
        inside = False
    return(inside)
#%%
def bbox_transpose(wormbbox, embryobbox):
    embryobbox[0] = embryobbox[0] - wormbbox[0]
    embryobbox[1] = embryobbox[1] - wormbbox[1]
    
    return(embryobbox)
#%%
def seg_transpose(wormbbox, embryo_seg):
    for y in range(0, len(embryo_seg[0])): #y is a single point within the segmentation coords
        embryo_seg[0][y] = embryo_seg[0][y] - wormbbox[y%2]
    return(embryo_seg)
#%%
'''
working_dir = "C:/Users/ebjam/Desktop/labeled/test/"
transposed_embryo_out = {"annotations": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}], "images": []}
wormcount = 0
with open("C:/Users/ebjam/Desktop/labeled/test.json") as test_json:
    test_set = json.load(test_json) #make big json readable object
    dump_dict = {"annotations": [], "images": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}]} #make empty dump dict
    hitno = 0 #set up hitno - used to numebr embryos to avoid duplicates for single embryos that appear in two worm boxes
    super_giant_embryo_list = [i for i in test_set["annotations"] if i["category_id"] == 1] # All embryos in every image
    for q in range(0, len(test_set["images"])): #for each image within the dataset
        im = cv2.imread(working_dir + test_set["images"][q]["file_name"]) #load the image with cv2
        imid = test_set["images"][q]["id"] #get the image ID to call all embryos and worms in the image
        #print("image#", q)
        #print("image id", imid)
        imagewormlist = [] #make empty list of worms for this image
        imageemblist = [] #make empty list of embryos for this image
        imagewormlist = [i for i in test_set["annotations"] if i["category_id"] == 2 and i["image_id"] == imid] #All worms just within an image
        for w in imagewormlist: #For each worm within the image
            one_worm_embryos = [i for i in super_giant_embryo_list if bbox_relation(w["bbox"], i["bbox"]) == True] #See if embryos are in bbox
            crop_bbox = w["bbox"] # Use worm bbox to crop input image
            cropim = im[math.floor(crop_bbox[1]):math.ceil(crop_bbox[1] + crop_bbox[3]), math.floor(crop_bbox[0]):math.ceil(crop_bbox[0]+crop_bbox[2])]
            cv2.imwrite(working_dir + test_set["images"][q]["file_name"][0:-4] + "_totalworm_" + str(wormcount) + ".png", cropim)
            keyname = test_set["images"][q]["file_name"][0:-4] + "_totalworm_" + str(wormcount) + ".png" #setting the save name as a variable to store
            imdict = {} #empty image dictionary to be made for newly cropped image
            imdict["height"] = crop_bbox[3]
            imdict["width"] = crop_bbox[2]
            imdict["id"] = wormcount #id = count of worms from 0 - n-1
            imdict["file_name"] = keyname
            dump_dict["images"].append(imdict)

            for e in range(0, len(one_worm_embryos)): #for embryo within worm bbox dimensions
                temp_store = {}
                temp_store["area"] = one_worm_embryos[e]["area"]
                temp_store["bbox"] = bbox_transpose(w["bbox"], one_worm_embryos[e]["bbox"]) #Transpose bbox (x min - x min, y min - y min)
                temp_store["category_id"] = one_worm_embryos[e]["category_id"]
                temp_store["id"] = hitno
                temp_store["image_id"] = imdict["id"]
                temp_store["iscrowd"] = 0
                temp_store["segmentation"] = seg_transpose(w["bbox"], one_worm_embryos[e]["segmentation"]) #Same transposition for segmentation
                hitno +=1 
                dump_dict["annotations"].append(temp_store)
                
            wormcount +=1
with open(working_dir + "worm_by_worm_embryo_test_transposed.json", "w") as outfile: #End of the "for j..." loop on line 59(or close to) - j iterates over worms
    json.dump(dump_dict, outfile, indent=4)      
'''
#%%
working_dir = "C:/Users/ebjam/Desktop/labeled/test/"
transposed_embryo_out = {"annotations": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}], "images": []}
wormcount = 0
hitno = 0 #set up hitno - used to numebr embryos to avoid duplicates for single embryos that appear in two worm boxes
dump_dict = {"annotations": [], "images": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}]} #make empty dump dict
with open("C:/Users/ebjam/Desktop/labeled/test.json") as test_json:
    test_set = json.load(test_json) #make big json readable object
    for q in range(0, len(test_set["images"])): #for each image within the dataset
        im = cv2.imread(working_dir + test_set["images"][q]["file_name"]) #load the image with cv2
        imid = test_set["images"][q]["id"] #get the image ID to call all embryos and worms in the image
        imageemblist = [a for a in test_set["annotations"] if a["category_id"] == 1 and a ["image_id"] == imid]
        imagewormlist = [b for b in test_set["annotations"] if b["category_id"] == 2 and b["image_id"] == imid] #All worms just within an image
        for w in imagewormlist: #For each worm within the image
             one_worm_embryos = [c for c in imageemblist if bbox_relation(w["bbox"], c["bbox"]) == True]