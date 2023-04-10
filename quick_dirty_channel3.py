# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:29:57 2023

@author: ebjam
"""
import os
import pickle
import numpy as np
import cv2
from matplotlib.path import Path

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

def go_fishing(todo, dy96_folder, threshval, fish):
    #If not predicting microsporidia, just return the todo with nothing added
    if fish == 0:
        return(todo)
    #Otherwise, I suppose we should predict microsporidia...
    elif fish == 1:
        image_no = 1
        print("Starting FISH thresholding")
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
                ctb_gimage = cv2.cvtColor(np.ascontiguousarray(cropped_to_bbox), cv2.COLOR_BGR2GRAY)
                retval, ctbg_thresh = cv2.threshold(ctb_gimage,threshval,255,cv2.THRESH_BINARY)
                seg_l = segmentation["transposed_segmentation"].tolist()
                seg_l.append(seg_l[0])
                #Create lists of x and y values
                xs, ys = zip(*seg_l) 
                height, width = ctbg_thresh.shape[:2]
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
                masked_image = ctbg_thresh.copy()
                # !!! Important !!! 
                # In worm but not infected = 0
                # Not in worm = 7
                # In worm and infected = 255
                masked_image[mask] = 7
                unique_values, value_counts = np.unique(masked_image, return_counts=True)
                # Create a dictionary of {value: count} pairs
                value_counts_dict = {value: count for value, count in zip(unique_values, value_counts)}
                #Deal with zero microsporidia
                if 255 not in value_counts_dict:
                    value_counts_dict[255] = 0
                total_px = value_counts_dict[0] + value_counts_dict[255]
                percent_infected = 100 * (value_counts_dict[255]/total_px)
                segmentation['fresh fish'] = ctbg_thresh
                segmentation['worm_area_px'] = total_px
                segmentation['percent_infected'] = percent_infected
                if worm_number % 2 == 0:
                  print("        Within image FISH " + str(100 * (worm_number/len(record["single_worms"])))[:5] + "% complete.")
                worm_number += 1
        return(todo)
    
#%%

todo = load_info("E:/2023-03-14/DY96/")

fish = go_fishing(todo, "E:/2023-03-14/DY96/", 35, 1)

#%%
filehandler =  open(os.path.join("E:/2023-03-14/DY96/" + "auto_fish.pickle"), "wb")
pickle.dump(fish, filehandler)
filehandler.close()