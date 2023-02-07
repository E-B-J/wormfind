# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:54:50 2023

@author: ebjam
"""
from ultralytics import YOLO
import os
import cv2
import pickle
#%%
def transpose_segmentation(bbox, segmentation, h, w):
    minx = bbox[0]
    miny = bbox[1]
    for i in segmentation[0]:
        '''
        i[0] * w turns decimal location into real location in input image. 
        i.e. coordinate of x = 0.567 on an image 1000 pixels wide would be pixel 567
        
        '- minx' and '-miny' transposes segmentation to the bbox rather than the full image
        '''
        i[0] = (i[0] * w) - minx
        i[1] = (i[1] * h) - miny
    #Segmentation is now transposed to bbox
    return(segmentation)
    
def run_worm_detection(inputfolder, model_path):
    model = YOLO(model_path)
    full_image_results = []
    img_list = [q for q in os.listdir(inputfolder) if q.endswith(".png")]
    annotationlist = {'worm_by_worm_annotations': [], 'info':{}}
    #Populate info list so it's possible to keep track of versions used for data generation
    annotationlist["info"]["model"] = model_path
    annotationlist["info"]["input_folder"] = inputfolder
    annotationlist["info"]["GUI"] = "not yet"
    for q in range(0, len(img_list)):
        title = img_list[q]
        img = cv2.imread("input folder" + title)
        h, w, c = img.shape
        results = model.predict(source = img)
        full_image_results.append(results[0])
        boxes = results[0].boxes
        masks = results[0].masks
        DY96img = cv2.imread(inputfolder + "DY96/" + title)#load dy96 image
        wormcount = 1
        #Make crops folder!
        for w in range(0, len(boxes)):
            #Make annotation file for single worm with transposed segmentation, original image bbox(image shape), and title 
            thisworm = {"name": "", "image": "", "worm_no": 0, "bbox": [], "wormsegmentation": []}
            #Get bbox, seg + trnasposed seg, and title
            bbox = boxes[w].xyxy
            segment = masks[w].segment
            transposed_segmentation = transpose_segmentation(bbox, segment, h, w)
            crop_title = inputfolder + "DY96_crops/" + title[:-4] + "_worm_" + str(wormcount) + ".png"
            #Add name, orig image, worm no, bbox, and seg to annotation file
            thisworm["name"] = crop_title
            thisworm["image"] = title
            thisworm["worm_no"] = wormcount
            thisworm["bbox"] = bbox
            thisworm["wormsegmentation"] = transposed_segmentation
            #Add 'thisworm' to annotationlist - made on line ~35
            annotationlist["worm_by_worm_annotations"].append(thisworm)
            #Crop and save the worm from DY96 image
            worm_crop = DY96img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(crop_title, worm_crop)
            #Increase wormcount
            wormcount += 1
        #Little thing to track progress
        if q % 5 == 0:
            percent_complete = 100 * (q/len(img_list))
            print("We're " + str(percent_complete) + "% through.")
    return(full_image_results, annotationlist)
#%%
#Run the model! Set inputs here!
inputpath = "/path/to/folder/with/DAPI/as/png/and/dy96/in/subfolder/"
pathtomodel = "/path/to/mode/file/"
full_image_results, worm_by_worm_results = run_worm_detection(inputpath, pathtomodel)

#Save the results!!
with open(inputpath + 'full_image_results.pickle', 'wb') as handle:
    pickle.dump(full_image_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(inputpath + 'DY96_crops/worm_by_worm_results.pickle', 'wb') as handle:
    pickle.dump(worm_by_worm_results, handle, protocol=pickle.HIGHEST_PROTOCOL)