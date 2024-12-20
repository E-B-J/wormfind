# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:54:50 2023

@author: ebjam

Currently probably only works on windows!

"""
from ultralytics import YOLO
import os
import cv2
import pickle
from shapely.geometry import Polygon


def size_filter(full_image_results, filter_size):
    """
    

    Parameters
    ----------
    full_image_results : Output from detector - can be before or after edge filter
    filter_size : size in um^3 used for filter. Filter is done with <= (less than or equal)

    Returns
    -------
    Size filtered results.

    """
    for result in full_image_results['results']:
        #print(len(result.masks.xyn))
        small_shape_index = []
        for q in range(0, len(result.masks.xyn)):
            mask = result.masks.xyn[q]
            tempmask = mask.copy()
            #print(tempmask)
            for i in tempmask:
                #Unnormalize to pixels, and scale to microns
                i[0] = (i[0] * 2752 * 1.1013)
                i[1] = (i[1] * 2208 * 1.1013)
                        #Tuple for polygon
            mask_for_poly = [tuple(x) for x in tempmask]
            #Close polygon
            if mask_for_poly[0] != mask_for_poly[len(mask_for_poly)-1]:
                mask_for_poly.append(mask_for_poly[0])
            area = (Polygon(mask_for_poly).area)

            if area <= filter_size:
                #Changes length of masks!! Need to do this to a temp holder!
                small_shape_index.append(q)
        #SSI contains indeces to remove
        #Make SSI go from high to low, then pop in order
        small_shape_index.sort(reverse=True)
        for w in range(0, len(small_shape_index)):
            result.masks.xyn.pop(small_shape_index[w])
        #print(len(result.masks.xyn))   
    return(full_image_results)    

def edge_filter(full_image_results, strictness = 0, edge_length = 1):
    """
    Parameters
    ----------
    full_image_results : YOLO predictions, can be before or after the size filter.
    
    strictness : optional - the extra number of rows in fromt he edge to filter.
        The default is 0, i.e. only objects with pixels in the absolute edge. 
        1 = edge two rows.
        2 = edge three rows
    
    edge_length: how many vertices on the edge to count as an edge case - default 1
    
    Returns
    -------
    filtered results

    """
    for result in full_image_results['results']:
        for q in range(0, len(result.masks.xyn)):
            mask = result.masks.xyn[q]
            tempmask = mask.copy()
            for i in tempmask:
                #Unnormalize to pixels, and scale to microns
                i[0] = (i[0] * 2752)
                i[1] = (i[1] * 2208)
            edge_case_indexes = []
            x_edge = []
            y_edge = []
            # Filter based on x coordinate and strictness
            if i[0] <= 0 + strictness:
                x_edge.append(i)
            elif i[0] >= 2752 - strictness:
                x_edge.append(i)
            # Filter based on y coordinate and strictness
            if i[1] <= 0 + strictness:
                y_edge.append(i)
            elif i[1] >= 2208 - strictness:
                y_edge.append(i)
            # Get number of vertices on the edge of image
            edge_cases = len(x_edge) + len(y_edge)
            # Record index of objects with too much edginess 
            if edge_cases <= edge_length:
                edge_case_indexes.append(q)
        
        # Reverse order sort the list of indexes - allows use to pop one after another without affecting important indexes
        edge_case_indexes.sort(reverse=True)
        # Pop
        for w in range(0, len(edge_case_indexes)):
            result.masks.xyn.pop(edge_case_indexes[w])

    return(full_image_results)
    
def transpose_segmentation(bbox, segmentation, h, w):
    # Transposes segmentation coordinates to bbox.
    minx = bbox[0]
    miny = bbox[1]
    for i in segmentation:
        '''
        YOLO returns normalized/decimal location (pixel 100 on a axis length 200 would have coordinate 0.5).
        Need to 'un normalized' before transposing.
        i[0] * w turns decimal location into real location in input image. 
        i.e. coordinate of x = 0.567 on an image 1000 pixels wide would be pixel 567
        
        '- minx' and '-miny' transposes segmentation to the bbox rather than the full image
        '''
        i[0] = (i[0] * w) - minx
        i[1] = (i[1] * h) - miny
    #Segmentation is now transposed to bbox
    return(segmentation)
    
def run_model(model, source):
    result = model.predict(source = source, classes = [1])
    return(result)

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
        results = run_model(model, img)
        #results = model.predict(source = img, classes = [1])
        #Save results, prep DY96 data for cropping -- should move to GUI!
        #!! Size filter here!!
        full_image_results.append(results[0])
        boxes = results[0].boxes
        masks = results[0].masks
        segments = masks.xyn
        DY96img = cv2.imread(inputfolder + "DY96/" + title[:-4] + "DY96.png")#load dy96 image
        wormcount = 1
        #Make crops folder!
        for e in range(0, len(boxes)):
            #Make annotation file for single worm with transposed segmentation, original image bbox(image shape), and title 
            thisworm = {"name": "", "image": "", "worm_no": 0, "bbox": [], "wormsegmentation": []}
            #Get bbox, seg + trnasposed seg, and title
            size_tester = segments[e]
            
            
            for i in size_tester:
                #Unnormalize to pixels, and scale to microns
                i[0] = (i[0] * 2752 * 1.1013)
                i[1] = (i[1] * 2208 * 1.1013)
            #Tuple for polygon
            mask_for_poly = [tuple(x) for x in size_tester]
            #Close polygon
            if mask_for_poly[0] != mask_for_poly[len(mask_for_poly)-1]:
                mask_for_poly.append(mask_for_poly[0])
            area = (Polygon(mask_for_poly).area)
            filter_size = 10000
            filtered_worms = 0
            #Only progress if worm is big enough!! Drops out maybe half the worms.
            if area > filter_size:  
                this_seg = segments[e]
                bbox = boxes[e].xyxy
                bbox = bbox.tolist()
                bbox_use = bbox[0]
                transposed_segmentation = transpose_segmentation(bbox_use, this_seg, h, w)
                crop_title = inputfolder + "DY96/" + title[:-8] + "DY96_worm_" + str(wormcount) + ".png"
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
            else:
                thisworm["filtered_worms"] = filtered_worms
                filtered_worms +=1
                
        #Little thing to track progress every 5 images
        if q % 5 == 0:
            percent_complete = 100 * (q/len(img_list))
            print("We're " + str(percent_complete) + "% through.")
        index_order = {'image_titles': img_list, 'results': full_image_results}
    return(index_order, annotationlist, img_list)
#%%
#Run the model! Set inputs here! TRAIL SLASH ON DIRECTORIES
inputpath = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/combo/input/png/test+val/full/DAPI/"
pathtomodel = "C:/Users/ebjam/Downloads/best.pt"
full_image_results, worm_by_worm_results, img_list = run_worm_detection(inputpath, pathtomodel)

size_filtered_results = size_filter(full_image_results, 15000)
edge_filtered_results = edge_filter(size_filtered_results, strictness = 25)

#Save the results!!
with open(inputpath + 'full_image_results2.pickle', 'wb') as handle:
    pickle.dump(edge_filtered_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(inputpath + 'worm_by_worm_results2.pickle', 'wb') as handle:
    pickle.dump(worm_by_worm_results, handle, protocol=pickle.HIGHEST_PROTOCOL)