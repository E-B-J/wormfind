# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:21:48 2022

@author: ebjam
"""
from pycocotools.coco import COCO
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import math

#%% Function to see if one bbox (embryo) is inside another (worm)
def bbox_relation(wormbbox, embryobbox):
    #if worm x (& y) is smaller than embryo x (&y), 
    if wormbbox[0] <= embryobbox[0] and wormbbox[1] <= embryobbox[1]:
        #If worm xmax is bigger than embryo xmax
        if (wormbbox[0] + wormbbox[2]) > (embryobbox[0] + embryobbox[2]):
            #If worm ymax is bigger than embryo ymax
            if (wormbbox[1] + wormbbox[3]) > (embryobbox[1] + embryobbox[3]):
                inside = True
            else:
                inside = False
        else:
            inside = False
    else:
        inside = False
    return(inside)
#%% Function to transpose bbox to cropped image with no padding
def bbox_transpose(wormbbox, embryobbox):
    embryobbox[0] = embryobbox[0] - wormbbox[0]#x = x-xmin
    embryobbox[1] = embryobbox[1] - wormbbox[1]#y = y-ymin
    
    return(embryobbox)
#%%Function to transpose seg to cropped image with no padding
def seg_transpose(wormbbox, embryo_seg):
    seg_holder = embryo_seg
    for y in range(0, len(seg_holder[0])): #y is a single point within the segmentation coords
        if y % 2 == 0:#if y is x coord
            seg_holder[0][y] -=wormbbox[0] 
        elif y % 2 == 1:#if y is y cord
            seg_holder[0][y] -=wormbbox[1]
    return(seg_holder)
#%%
'''Big loop to take input images - crop the worm bboxes from them, 
and transpose worm and embryo annotations to these new cropped images.
Annotation ID duplicates are not yet dealt with, 
presumably just aother loop at the end of things will sort it out

Currently, I think it only needs working dir and annFile as variables,
     - possible to make into a functionreturning the dump dictionary?
'''

#Directory where files will be dumped, and where imput images are
working_dir = "C:/Users/ebjam/Desktop/labeled/test/"
#Annotation file for images
annFile = "C:/Users/ebjam/Desktop/labeled/test.json"
#dump dict looks like a java object, but it really a python dictionary we're going to be saving at the end
dump_dict = {"image": [], "categories": [], "annotations": [], }
coco=COCO(annFile) #load via cocoapi
cats = coco.loadCats(coco.getCatIds()) #get categorys to add to each json out
dump_dict["categories"] = cats
catIds = coco.getCatIds(catNms=['worm']) #use to pull cat ID from cat
imgIds = coco.getImgIds(catIds=catIds ) #get all images with the specified category above in
new_im_id = 0
for q in range(0, len(imgIds)):#q is an image with interesting category in it
    img = coco.loadImgs(imgIds[q])[0]#load image name via cocoapi
    I = cv2.imread(working_dir + img['file_name']) #load the image with cv2
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) #Get image worm annotations
    anns = coco.loadAnns(annIds)
    embryo_annIds = coco.getAnnIds(imgIds=img['id'], catIds=[1], iscrowd=None) #Get image embryo annotations
    embryo_anns = coco.loadAnns(embryo_annIds)
    for worm_no in range(0, len(anns)):#worm_no is zero indexed #worms in image
        #dump dict looks like a java object, but it really a python dictionary we're going to be saving at the end
        #dump_dict = {"image": [], "categories": [], "annotations": [], }
        #Get the bbox from the worm to use as a crop box
        crop_bbox = anns[worm_no]["bbox"]
        #Get a list of embryos that are inside this crop box 
        embryos_in_worm = [embryo_anns[a] for a in range(0, len(embryo_anns)) if bbox_relation(crop_bbox, embryo_anns[a]['bbox']) == True]
        #make cropped image
        cropim = I[math.floor(crop_bbox[1]):math.ceil(crop_bbox[1] + crop_bbox[3]), math.floor(crop_bbox[0]):math.ceil(crop_bbox[0]+crop_bbox[2])]
        #save cropped image
        file_name_variable = img["file_name"] + "_worm_no_" + str(worm_no)
        cv2.imwrite(working_dir + file_name_variable, cropim)
        for b in range(0, len(embryos_in_worm)): #b is an embryo within the cropped image
            #Transpose embryo seg FIRST ALWAYS!! (function above)
            embryos_in_worm[b]['segmentation'] = seg_transpose(crop_bbox, embryos_in_worm[b]['segmentation'])
            #Transpose embryo bbox (function above - could also calculate from seg)
            embryos_in_worm[b]['bbox'] = bbox_transpose(crop_bbox, embryos_in_worm[b]['bbox'])
            #add new image id(assigning to image after!)
            embryos_in_worm[b]['image_id'] = new_im_id
            #add embryo annotation to dump_dict_annotations
            dump_dict["annotations"].append(embryos_in_worm[b])
        #Same but worm - only one per image probably, so no need for loop.
        worm_transpose_store = anns[worm_no]
        worm_transpose_store["image_id"] = new_im_id
        worm_transpose_store["segmentation"] = seg_transpose(anns[worm_no]["bbox"], worm_transpose_store["segmentation"])
        worm_transpose_store["bbox"] = bbox_transpose(anns[worm_no]["bbox"], anns[worm_no]["bbox"])
        dump_dict["annotations"].append(worm_transpose_store)
        #updating image dictionary for new json
        img_transpose_store = img
        img_transpose_store["height"] = crop_bbox[3]
        img_transpose_store["width"] = crop_bbox[2]
        img_transpose_store["id"] = new_im_id #id = count of worms from 0 - n-1
        img_transpose_store["file_name"] = file_name_variable
        #add image dictionary to dump_dict
        dump_dict["image"].append(img_transpose_store)
        #Update image number
        new_im_id += 1

#Straighten out any annotation duplications due to bbox overlap
for w in range(0, len(dump_dict["annotations"])):
    dump_dict['annotations'][w]["id"] = w
    
#Dump a single coco format json with all cropped images and annotations inside.
with open(working_dir + "worm_by_worm_embryo_test_transposed.json", "w") as outfile:
    json.dump(dump_dict, outfile, indent=4)   