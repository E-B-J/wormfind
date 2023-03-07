# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:43:13 2023

@author: ebjam
"""
# ML imports
from ultralytics import YOLO
from skimage import feature, future #RF tools
# Image/array handling imports
import cv2
from shapely.geometry import Point, Polygon
from PIL import Image, ImageDraw
#import rasterio.features
import numpy
import numpy.ma as ma
# Data/object handling imports
import os
import numpy as np
from functools import partial #For feature function
import pickle
import csv



#model_string = "C:/Users/ebjam/Downloads/tile_embryo_detect_l_20230206.pt"
#input_folder = ""
#dy96_folder = os.joinpath(input_folder, "DY96")
#%% Defining functions
'''
Workflow: For file in .result
for each file load DY96 image. For each worm segmentation, load transposed worm seg and bbox, temp crop to bbox, and run detectors


'''
# Defining the detectors
"""
Created on Mon Feb  6 11:43:13 2023

@author: ebjam
"""
# ML imports
from ultralytics import YOLO
from skimage import feature, future #RF tools
# Image/array handling imports
import cv2
from shapely.geometry import Point, Polygon
from PIL import Image, ImageDraw
from matplotlib.path import Path
#import rasterio.features
import numpy
import numpy.ma as ma
# Data/object handling imports
import os
import numpy as np
from functools import partial #For feature function
import pickle
import csv

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

def predict_embryos(todo, embryo_model_path, dy96_folder, embryos):
    #If not predicting embryos, just return the todo with nothing added
    if embryos == 0:
        print("No embryo detection selected.")
        return(todo)
    #Otherwise, I suppose we should predict embryos...
    elif embryos == 1:
        print("Starting embryo detection.")
        #Load model
        model = YOLO(embryo_model_path)
        # For image in imput images, load it's result file
        image_no = 1
        for record in todo:
            print("    Working on image " + str(image_no) + " of " + str(len(todo)))
            image_no +=1
            # Load the DY96 image to crop and analyze
            input_image = record['input_image']
            dy96_image_title = input_image[:-8] + "DY96.png"
            dy96_path = os.path.join(dy96_folder, dy96_image_title)
            dy96_image = cv2.imread(dy96_path)
            #Now iterate over predicted worms
            worm_number = 1
            for segmentation in record['single_worms']:
                # Crop in to specific worm bbox
                bbox = segmentation['bbox']
                cropped_to_bbox = dy96_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                # Predict embryos
                results = model.predict(source = np.ascontiguousarray(cropped_to_bbox), save=False, save_txt=False)
                # Get embryo centerpoints
                centerlist = find_centers(results[0].boxes)
                # Add both to dict
                segmentation['embryo_bboxes'] = results[0].boxes
                segmentation['embryo_centers'] = centerlist
                # Check worm segmentation for embryos
                # Load segmentation, make sure it's closed
                transposed_seg = segmentation['transposed_segmentation']
                # Convert segmentation into a list of tuples to plot polygon.
                seg_for_poly = [tuple(x) for x in transposed_seg]
                if seg_for_poly[0] != seg_for_poly[len(seg_for_poly)-1]:
                    seg_for_poly.append(seg_for_poly[0])
                # Make polygon out of worm segmentation
                polygon = Polygon(seg_for_poly)
                internal_embryos = []
                listarray = []
                for point in centerlist:
                    # If the point is inside the worm, then add the centerpoint to the internal embryo list.
                    if polygon.contains(point):
                        listarray.append([point.x, point.y])
                        internal_embryos.append(point)
                segmentation['internal_embryo_points'] = internal_embryos
                segmentation['internal_embryo_centers'] = listarray
                segmentation['#_internal_embryos'] = len(internal_embryos)
                if worm_number % 2 == 0:
                    print("        Within image embryo detection " + str(100 * (worm_number/len(record["single_worms"])))[:5] + "% complete.")
                worm_number += 1
        return(todo)

# Microsporidia - debugged up to prediction

sigma_max = 16
sigma_min = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=False,
                        sigma_min=sigma_min, sigma_max=sigma_max)

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

# Saving results
def csv_saver(embryos, microsporidia, save_csv, finaldict, input_folder):
    if save_csv == 1:
        if embryos == 0 and microsporidia == 0:
            print("No predictions saved as no predictions generated, check 'embryos' or 'microsporidia' in the detector are set to 1 rather than default of 0.")
        else:
            print("Saving results to csv within input/DY96 folder.")
            # Open csv to save stuff in
            savefile = open(os.path.join(input_folder,'demo_file.csv'), 'w', newline='')
            # Make writer write results to that save file
            writer = csv.writer(savefile)
            # Get column headers by getting list of by worm dictionary keys
            column_heads = list(finaldict[0]["single_worms"][0].keys())
            # Write those headers to the first line of the file
            writer.writerow(column_heads)
            # Write values from each worm into csv
            for image in finaldict:
                for worm in image["single_worms"]:
                    writer.writerow(worm.values())
            # Close file after writing
            savefile.close()
            print("Saving complete.")
    return()

# All-in-one detector with option handling

def detector(inputfolder, 
             # Prediction selection
             embryos = 0, microsporidia = 0, 
             # Save selection
             save_csv = 1, save_pickle = 1, 
             # Model path definition
             embryo_model = "/content/drive/MyDrive/yolov8_out/embryo_detect_tile_l/detect/train/weights/tile_embryo_detect_l_20230206.pt",
             microsporidia_model = "/content/drive/MyDrive/rf_models/100trees10branches_just_intensity.pickle"
             ):
    # Quick empty path throwback - shouldn't trigger with the default model paths!
    if len(embryo_model) < 3:
        print("Please input/check embryo model path")
        return()
    if len(microsporidia_model) < 3:
        print("Please input/check microsporidia model path")
        return()
    # With how I wrote this, must make embryo detection !THEN! microssporidia/other channel. Other orders won't work as is.

    todo = load_info(inputfolder)
    todo_with_embryos = predict_embryos(todo, embryo_model, inputfolder, embryos)
    todo_with_microsporidia = predict_microsporidia(todo_with_embryos, microsporidia_model, inputfolder, microsporidia)
    csv_saver(embryos, microsporidia, save_csv, todo_with_microsporidia, inputfolder)
    # Save results as pickle for expanded datause
    print("Pickling results in input folder")
    filehandler =  open(os.path.join(inputfolder + "predictions.pickle"), "wb")
    pickle.dump(todo_with_microsporidia, filehandler)
    filehandler.close()
    print("Pickle complete.")
    return(todo_with_microsporidia)
#%%
res = detector(inputfolder = "C:/Users/ebjam/Downloads/gui_testers-20230213T211340Z-001/second_detector_testers_96/DY96/", embryos = 1, microsporidia = 0)
