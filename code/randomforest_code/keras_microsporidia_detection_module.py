# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:41:57 2023

@author: ebjam
"""
#%% Imports
import os
import pickle
import cv2
from shapely.geometry import Point
from PIL import Image, ImageDraw
import numpy.ma as ma
import numpy as np
import 
#%% Def utils
def load_info(input_folder):
    # Empty list to contain all loaded resut files
    todo = []
    # Load all files out of the gui
    result_list = [q for q in os.listdir(input_folder) if q.endswith(".result")]
    
    for result_file in result_list:
        # Load result file per image
        segmentation_record = os.path.join(input_folder, result_file)
        file = open(segmentation_record,'rb')
        seg_record = pickle.load(file)
        todo.append(seg_record)
    return(todo)

# Embryo detection - debugged & working!!

def find_centers(theboxes):
    centerpoints = []
    for box in theboxes:
        xyxy = box.xyxy.tolist()
        xyxy = xyxy[0]
        cx = (xyxy[0] + xyxy[2])/2
        cy = (xyxy[1] + xyxy[3])/2
        cp = Point(cx, cy)
        centerpoints.append(cp)
    return(centerpoints)


#%%

def predict_microsporidia(todo, microsporidia_model_path, dy96_folder, microsporidia):
    #If not predicting microsporidia, just return the todo with nothing added
    if microsporidia == 0:
        return(todo)
    #Otherwise, I suppose we should predict microsporidia...
    elif microsporidia == 1:
        #Load model
        file = open(microsporidia_model_path,'rb')
        clf = pickle.load(file)
        # For image in imput images, load it's result file
        image_no = 1
        print("Starting microsporidia prediction")
        for record in todo:
            input_image = record['input_image']
            dy96_image_title = input_image[:-8] + "DY96.png"
            dy96_path = os.path.join(dy96_folder, dy96_image_title)
            dy96_image = cv2.imread(dy96_path)
            print("    Working on image no " + str(image_no) + " of " + str(len(todo)))
            image_no += 1
            worm_number = 1
            for segmentation in record['single_worms']:
                # Crop in to specific worm bbox
                bbox = segmentation['bbox']
                cropped_to_bbox = dy96_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                gimage = cv2.cvtColor(np.ascontiguousarray(cropped_to_bbox), cv2.COLOR_BGR2GRAY)
                gimage_features = features_func(gimage)
                # Predict microsporidia
                pred = future.predict_segmenter(gimage_features, clf)
                del gimage_features
                pred[pred == 1] = 0
                pred[pred == 2] = 255
                seg_l = segmentation["transposed_segmentation"].tolist()
                seg_l.append(seg_l[0])
                #Create lists of x and y values
                xs, ys = zip(*seg_l) 
                height, width = pred.shape[:2]
                # Create a mesh grid representing the image dimensions
                x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

                # Convert the coordinates to a 1D array
                x_coords = x_grid.reshape(-1)
                y_coords = y_grid.reshape(-1)

                # Create a path for the polygon
                polygon_path = Path(np.column_stack([xs, ys]))

                # Check which points are inside the polygon
                points_inside_polygon = polygon_path.contains_points(np.column_stack([x_coords, y_coords]))

                # Reshape the boolean array to match the image dimensions
                mask = points_inside_polygon.reshape((height, width))
                # Invert the mask - want pixels inside worm, not outside!
                mask = ~mask
                #Mask image
                masked_image = pred.copy()
                masked_image[mask] = np.nan
                unique_values, value_counts = np.unique(masked_image, return_counts=True)
                # Create a dictionary of {value: count} pairs
                value_counts_dict = {value: count for value, count in zip(unique_values, value_counts)}
                #Deal with zero microsporidia
                if 255 not in value_counts_dict:
                    value_counts_dict[255] = 0
                total_px = value_counts_dict[0] + value_counts_dict[255]
                percent_infected = 100 * (value_counts_dict[255]/total_px)
                segmentation['pred'] = pred
                segmentation['worm_area_px'] = total_px
                segmentation['percent_infected'] = percent_infected
                if worm_number % 2 == 0:
                  print("        Within image microsporidia prediction " + str(100 * (worm_number/len(record["single_worms"])))[:5] + "% complete.")
                worm_number += 1
        return(todo)