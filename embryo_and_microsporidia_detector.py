# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:43:13 2023

@author: ebjam
"""
#ML imports
from ultralytics import YOLO
from skimage import feature, future, segmentation #RF tools
from sklearn.ensemble import RandomForestClassifier # The RF itself
#Image/array handling imports
import cv2
from shapely.geometry import Point, Polygon
import rasterio.features
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy
import numpy.ma as ma
import seaborn as sns
#Data/object handling imports
import os
from functools import partial #For feature function
import pickle
import csv
import json
#Tracability import
from datetime import date

#Get today's date
today = date.today()

model_string = "/path/to/model/"
input_folder = "path/to/folder/full/of/cropped/DY96worms" 
#%%Defining functions


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
        for worm in worm_seg_and_egg:
            if worm["name"] == key:
                worm["embryo_bboxes"] = results[0].boxes
                worm["embryo_centers"] = find_centers(results[0].boxes)
    return(worm_segmentations)

def check_gravidity(worm_seg_and_egg, save_img = 0):
    for worm in worm_seg_and_egg:
        name = worm["name"]
        seg = worm['wormsegmentation']
        embryo_centerpoints = worm["embryo_centers"]
        seg_for_poly = [tuple(x) for x in seg]
        polygon = Polygon(seg_for_poly)
        for center_point in embryo_centerpoints:
            if polygon.contains(center_point):
                if "embryo_no" not in worm:
                    # Make keypoint list
                    worm["embryo_no"] = 0
                    worm["gravidity"] = 0
                    worm["internal_embryos"] = []
                    # Add segmentation to keypoint list, have to add 'v' coord (x, y, v):
                    # 0 = not labelled, 1 = labeled not visable, 2 = labelled and visible
                worm['embryo_no'] += 1
                worm["internal_embryos"].append(center_point)
        if worm["embryo_no"] > 0:
            worm["gravidity"] = 1
        if save_img == 1:
            #Load image, and plot centerpoint of each embryo as large marker!
            xpoints = []
            ypoints = []
            for i in worm["internal_embryos"]:
                xpoints.append(i[0])
                ypoints.append(i[1])
            img = plt.imread(worm['name'])
            plt.imshow(img)
            plt.plot(xpoints, ypoints, marker = '*', color = 'r',
                     markersize=15, markeredgecolor = 'k')
            plt.savefig(name[:-4] +"embryos.png")
    return(worm_seg_and_egg)

sigma_max = 16
sigma_min = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=False,
                        sigma_min=sigma_min, sigma_max=sigma_max)

def predict_microsporidia(id_and_array, worm_egg_and_seg, save_img = 0):
    file = open("C:/Users/ebjam/Documents/GitHub/wormfind/100trees10branches_just_intensity.pickle",'rb')
    clf = pickle.load(file)
    #Generate mask from worm segementation to define area of image we're interested in.
    for worm in worm_egg_and_seg['worm_by_worm_annotations']:
        #load worm seg, bbox, and name
        seg = worm["wormsegmentation"]
        bbox = worm["bbox"]
        image_title = worm["name"]
        #Turn seg into polygon - first turn into a list of tuples, then make sure it's closed
        seg_for_poly = [tuple(x) for x in seg]
        if seg_for_poly[0] != seg_for_poly[len(seg_for_poly)]:
            close_the_polygon = seg_for_poly[len(seg_for_poly)]
            seg_for_poly.append(close_the_polygon)
        poly = Polygon(seg_for_poly)
        #Now convert the polygon to a mask within it's bbox
        poly_mask= rasterio.features.rasterize([poly], out_shape=(((bbox[2]-bbox[0]), (bbox[3]-bbox[1]))))
        #Load dy96 image as greyscale, extract features, and run predicition
        image = cv2.imread(image_title)
        gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gimage_features = features_func(gimage)
        del gimage
        pred = future.predict_segmenter(gimage_features, clf)
        del gimage_features
        #Scale pred between 0 and 1 - means the mean of masked pred will be % infected
        pred[pred == 1] = 0
        pred[pred == 2] = 1
        masked_to_worm = ma.masked_array(pred, mask=poly_mask)
        del pred
        if save_img == 1:
            plt.imsave(image_title[:-4] + "pred.png", masked_to_worm, cmap="gray")
        #Length of masked array is the number of pixels^2 a worm is - can convert to specific microscope
        worm_area_px = len(masked_to_worm.flatten())
        percent_area = masked_to_worm.mean()
        #Add metric to datafile
        worm["percent_area"] = percent_area
        worm["area_px"] = worm_area_px
    return(worm_egg_and_seg)
#%% Building all-in-one function with option handling
def detector(inputfolder, embryos = 0, microsporidia = 0, saveimg = 0, save_csv = 1, save_pickle = 1):
    
    id_and_array, worm_segments = list_out_images(inputfolder)
    saveimg
    
    if embryos == 1:
        print("Predicting embryos...")
        worm_segments = check_gravidity(predict_embryos(id_and_array, worm_segments))
        print("Done predicting embryos")
    if microsporidia == 1:
        print("Predicting microsporidia")
        worm_segments = predict_microsporidia(id_and_array, worm_segments, save_img = saveimg)
        print("Done predicting microsporidia")
    #Line writer to handle specific outputs
    print("Saving results")
    if embryos == 1:
        if microsporidia == 1:
            with open(inputfolder + today + 'predictions_e_ms.csv', mode='w') as res_file:
                fieldnames = ['worm', 'gravid', 'embryos', 'area', '%area']
                worm_writer = csv.DictWriter(res_file, fieldnames=fieldnames)
                worm_writer.writeheader()
                for nworm in range(0, len(worm_segments['worm_by_worm_annotations'])):
                    worm_writer.writerow({'worm': worm_segments['worm_by_worm_annotations'][nworm]['name'],
                                     'gravid': worm_segments['worm_by_worm_annotations'][nworm]['gravidity'],
                                     'embryos': worm_segments['worm_by_worm_annotations'][nworm]['embryo_no'],
                                     'area': worm_segments['worm_by_worm_annotations'][nworm]['area_px'],
                                     '%area': worm_segments['worm_by_worm_annotations'][nworm]['percent_area']})
        elif microsporidia == 0:
            with open(inputfolder + today + 'predictions_e.csv', mode='w') as res_file:
                fieldnames = ['worm', 'gravid', 'embryos']
                worm_writer = csv.DictWriter(res_file, fieldnames=fieldnames)
                worm_writer.writeheader()
                for nworm in range(0, len(worm_segments['worm_by_worm_annotations'])):
                    worm_writer.writerow({'worm': worm_segments['worm_by_worm_annotations'][nworm]['name'],
                                     'gravid': worm_segments['worm_by_worm_annotations'][nworm]['gravidity'],
                                     'embryos': worm_segments['worm_by_worm_annotations'][nworm]['embryo_no']})
    elif embryos == 0 and microsporidia == 1:
        with open(inputfolder + today + 'predictions_ms.csv', mode='w') as res_file:
            fieldnames = ['worm', 'area', '%area']
            worm_writer = csv.DictWriter(res_file, fieldnames=fieldnames)
            worm_writer.writeheader()
            for nworm in range(0, len(worm_segments['worm_by_worm_annotations'])):
                worm_writer.writerow({'worm': worm_segments['worm_by_worm_annotations'][nworm]['name'],
                                     'area': worm_segments['worm_by_worm_annotations'][nworm]['area_px'],
                                     '%area': worm_segments['worm_by_worm_annotations'][nworm]['percent_area']})
    else:
        print("No predictions run, no results saved. Make sure you set either 'embryos' or 'microsporidia' to '1' to run prediction.")
    print("CSV saved")
    #Save results as pickle for expanded datause
    filehandler =  open(inputfolder + today + "predictions.pickle")
    pickle.dump(worm_segments,filehandler)
    filehandler.close()
    print("Pickle saved.")
#%%Run the thing

res = detector("input/folder/with/worm/predictions/and/cropped/dy96/", embryos = 1)

