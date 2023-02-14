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
    for i in segmentation:
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
    #Load model
    model = YOLO(model_path)
    #Initiate result list
    full_image_results = []
    #Make list of png files
    img_list = [q for q in os.listdir(inputfolder) if q.endswith(".png")]
    #Initiate annotation list
    annotationlist = {'worm_by_worm_annotations': [], 'info':{}}
    #Populate info list so it's possible to keep track of versions used for data generation
    annotationlist["info"]["model"] = model_path
    annotationlist["info"]["input_folder"] = inputfolder
    annotationlist["info"]["GUI"] = "not yet"
    #Iterate over images. q is image index in img list.
    for q in range(0, len(img_list)):
        #Get img and info
        title = img_list[q]
        img = cv2.imread(inputfolder + title)
        h, w, c = img.shape
        #Run prediction, only outputting worm class
        results = model.predict(source = img, classes = [1])
        #Save results, prep DY96 data for cropping -- should move to GUI!
        full_image_results.append(results[0])
        boxes = results[0].boxes
        masks = results[0].masks
        segments = masks.segments
        DY96img = cv2.imread(inputfolder + "DY96/" + title)#load dy96 image
        wormcount = 1
        #Make crops folder!
        for e in range(0, len(boxes)):
            #Make annotation file for single worm with transposed segmentation, original image bbox(image shape), and title 
            thisworm = {"name": "", "image": "", "worm_no": 0, "bbox": [], "wormsegmentation": []}
            #Get bbox, seg + trnasposed seg, and title
            bbox = boxes[e].xyxy
            bbox = bbox.tolist()
            bbox_use = bbox[0]
            this_seg = segments[e]
            transposed_segmentation = transpose_segmentation(bbox_use, this_seg, h, w)
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
            worm_crop = DY96img[int(bbox_use[1]):int(bbox_use[3]), int(bbox_use[0]):int(bbox_use[2])]
            cv2.imwrite(crop_title, worm_crop)
            #Increase wormcount
            wormcount += 1
        #Little thing to track progress
        if q % 5 == 0:
            percent_complete = 100 * (q/len(img_list))
            print("We're " + str(percent_complete) + "% through.")
        index_order = {'image_titles': img_list, 'results': full_image_results}
    return(index_order, annotationlist, img_list)
#%%
#Run the model! Set inputs here!
inputpath = "C:/Users/ebjam/Downloads/gui testers-20230213T211340Z-001/gui testers/"
pathtomodel = "C:/Users/ebjam/Downloads/best.pt"
full_image_results, worm_by_worm_results, img_list = run_worm_detection(inputpath, pathtomodel)

#Save the results!!
with open(inputpath + 'full_image_results.pickle', 'wb') as handle:
    pickle.dump(full_image_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(inputpath + 'DY96_crops/worm_by_worm_results.pickle', 'wb') as handle:
    pickle.dump(worm_by_worm_results, handle, protocol=pickle.HIGHEST_PROTOCOL)