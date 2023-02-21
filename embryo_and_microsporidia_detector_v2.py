# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:43:13 2023

@author: ebjam
"""
# ML imports
from ultralytics import YOLO
from skimage import feature, future, segmentation #RF tools
from sklearn.ensemble import RandomForestClassifier # The RF itself
# Image/array handling imports
import cv2
from shapely.geometry import Point, Polygon
import rasterio.features
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy
import numpy.ma as ma
import seaborn as sns
# Data/object handling imports
import os
from functools import partial #For feature function
import pickle
import csv
import json


model_string = "/path/to/model/"
input_folder = "path/to/folder/full/of/cropped/DY96worms" 
#%% Defining functions

#Pre-prediction

def load_model(model_string):
    model = YOLO(model_string)
    return(model)

def list_out_images(input_folder):
    # Returns a dictionary where the keys are image names, and the values are the worm segmentations as a list
    result_list = [q for q in os.listdir(input_folder) if q.endswith(".result")]
    for result_file in result_list:
        # Load result file per image
        segmentation_record = os.path.join(input_folder, result_file)
        file = open(segmentation_record,'rb')
        seg_record = pickle.load(file)
        # Get list of crops from image
        image_list = []
        seg_list = []
        for w in range(0, len(seg_record["single_worms"])):
            # Add ".png" to the end of save title in order to load the image in a second
            image_list.append(seg_record["single_worms"][w]["wormID"] + ".png")
            seg_list.append(seg_record["single_worms"][w]["transposed_segmentation"])
        # Add DY96 folder to beginning of image handle to allow loading.
        for image in image_list:
            image = os.joinpath(input_folder, image)
        # Zip together image name and transposed segmentation - allow loading of single crop image at a time rather than making a list of all as an array..
        crop_seg_dict = dict(zip(image_list, seg_list))
    return(crop_seg_dict)


# Embryo detection

def find_centers(theboxes):
    centerpoints = []
    for box in theboxes:
        x1, y1, x2, y2 = box.xyxy
        cx = (x1 + x2)/2
        cy = (y1 + y2)/2
        cp = Point(cx, cy)
        centerpoints.append(cp)
    return(centerpoints)

def predict_embryos(crop_seg_dict, model_string, embryo_prediction):
    # Predicts embryos, and checks if they're inside a predicted worm segmentation.
    # Init dictionary that will hold results 
    pass_between_dict = {"worm_by_worm": []}
    imlist = list(crop_seg_dict.keys())
    # Only load model if we're predicting!
    if embryo_prediction == 1:
        print("Prediciting embryos - loading model")
        model = load_model(model_string)
    # For image in list, load the image and predict embryos on full image
    for im in imlist:
        # If we want to spend some energy predicting embryos
        if embryo_prediction == 1:
            im_as_array = cv2.imread(im)
            results = model.predict(source = im_as_array, save=False, save_txt=False)
            # Get centerpoints of those embryos to see if they're in the worm
            centerlist = find_centers(results[0].boxes)
            worm = crop_seg_dict[im]
            # Make sure worm polygon is a closed polygon - i.e. is capable of having internal points.
            if worm[0] != worm[-1]:
                worm.append(worm[0])
            # Convert segmentation into a list of tuples to plot polygon.
            seg_for_poly = [tuple(x) for x in worm]
            # Make polygon out of worm segmentation
            polygon = Polygon(seg_for_poly)
            # Test to see if embryo centerpoint is inside the worm polygon - embryo centerpoints are already 'Points'
            internal_embryos = []
            for point in centerlist:
                # If the point is inside the worm, then add the centerpoint to the internal embryo list.
                if polygon.contains(point):
                    internal_embryos.append(point)
            # 'this_annotation' contains all current info about this anno - image name, worm seg, embryos
            this_annotation = {"image": im, "worm_seg": worm, "embryo_prediction": "yes", "all_embryos": centerlist, "internal_embryos": internal_embryos}
            # Add image name, worm segmentation, list of all centerpoints, and list of internal centerpoints to some megadict
            pass_between_dict["worm_by_worm"].append(this_annotation)

        elif embryo_prediction == 0:
            print("No embryo prediction selected.")
            worm = crop_seg_dict[im]
            if worm[0] != worm[-1]:
                worm.append(worm[0])
            this_annotation = {"image": im, "worm_seg": worm, "embryo_prediction": "no"}
            pass_between_dict["worm_by_worm"].append(this_annotation)
    # If we predict embryos or not, we retuen a pass between dictionary with image name as a path, and worm segmentation
    # Pass between dict will also be used to access images for microsporidia results
    # Microsporidia prediction = 0 return passbetween as is.
    print("Embryo detection complete.")
    return(pass_between_dict)


# Microsporidia detection

sigma_max = 16
sigma_min = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=False,
                        sigma_min=sigma_min, sigma_max=sigma_max)

def predict_microsporidia(pass_between_dict, model_path, microsporidia_prediction):
    # If we're predicting, load RF model to ID microsporidia
    if microsporidia_prediction == 1:
        print("Predicting microsporidia")
        file = open(model_path,'rb')
        clf = pickle.load(file)
        # Lots if image as array loading and modification - delete arrays when the're done to free up some space?
        # Load single image from images in pass_between_dict. Each has own annotation in "worm_by_worm"
        for q in range(0, len(pass_between_dict["worm_by_worm"])):
            # Get image name
            crop_image_title = pass_between_dict["worm_by_worm"][q]["image"]
            crop_image = cv2.imread(crop_image_title)
            # Make image greyscale
            gimage = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            # Generate features from gimage
            gimage_features = features_func(gimage)
            # Predict microsporidia
            pred = future.predict_segmenter(gimage_features, clf)
            del gimage_features
            # Scale pred between 0 and 1 - means the mean of masked pred will be % infected
            pred[pred == 1] = 0
            pred[pred == 2] = 1
            # Pred is prediction on the full cropped DY96 image - we want to mask it to just the worm area.
            # Load segmentation
            seg = pass_between_dict["worm_by_worm"][q]["worm_seg"]
            # Convert segmentation to something I can mask with
            seg_for_poly = [tuple(x) for x in seg]
            # Make sure polygon is closed, then make polygon
            if seg_for_poly[0] != seg_for_poly[len(seg_for_poly)]:
                close_the_polygon = seg_for_poly[len(seg_for_poly)]
                seg_for_poly.append(close_the_polygon)
            poly = Polygon(seg_for_poly)
            # Now convert the polygon to a mask within it's bbox
            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny
            poly_mask = rasterio.features.rasterize([poly], out_shape=((height, width))) #!!! Check the width and height order here!
            # Make masked prediction
            masked_to_worm = ma.masked_array(pred, mask=poly_mask)
            # Length of masked predition = number of pixels in worm = area in px^2 - should add this earlier!!
            worm_area_px = len(masked_to_worm.flatten())
            # Mean = percent infected as infected pixel = 1, and uninfected = 0
            percent_area = masked_to_worm.mean()
            pass_between_dict["worm_by_worm"][q]["px_area"] = worm_area_px
            pass_between_dict["worm_by_worm"][q]["percent_microsporidia"] = percent_area
            pass_between_dict["worm_by_worm"][q]["masked_microsporidia"] = masked_to_worm
            pass_between_dict["worm_by_worm"][q]["microsporidia_prediction"] = "yes"
    # If no prediction, return pass_between_dict with an add on in each worm that no microsporidia prediction took place.
    elif microsporidia_prediction == 0:
        print("Selected no microsporidia detection.")
        for q in range(0, len(pass_between_dict["worm_by_worm"])):
            pass_between_dict["worm_by_worm"][q]["microsporidia_prediction"] = "no"
    # Give back the dictionary with microsporidia prediction info
    print("Microsporidia deteection complete.")
    return(pass_between_dict)

# Saving results

def csv_saver(embryos, microsporidia, save_csv, finaldict, input_folder):
    if save_csv == 1:
        if embryos == 0 and microsporidia == 0:
            print("No predictions saved, as no predictions generated, check 'embryos' or 'microsporidia' in the detector are set to 1 rather than default of 0.")
        else:
            print("Saving results to csv.")
            # Open csv to save stuff in
            savefile = open(os.joinpath(input_folder,'demo_file.csv'), 'w')
            # Make writer write results to that save file
            writer = csv.writer(savefile)
            # Get column headers by getting list of by worm dictionary keys
            column_heads = list(finaldict["worm_by_worm"][0].keys())
            # Write those headers to the first line of the file
            writer.writerow(column_heads)
            # Write values from each worm into csv
            for worm in finaldict["worm_by_worm"]:
                writer.writerow(worm.values())
            # Close file after writing
            savefile.close()
            print("csv saving complete.")
    return()


# All-in-one detector with option handling

def detector(inputfolder, 
             # Prediction selection
             embryos = 0, microsporidia = 0, 
             # Save selection
             save_csv = 1, save_pickle = 1, 
             # Model path definition
             embryo_model = "",
             microsporidia_model = "C:/Users/ebjam/Documents/GitHub/wormfind/100trees10branches_just_intensity.pickle"
             ):
    # Quick empty path throwback - shouldn't trigger with the default model paths!
    if len(embryo_model) < 3:
        print("Please input embryo model path")
        return()
    if len(microsporidia_model) < 3:
        print("Please input microsporidia model path")
        return()
    # With how I wrote this, must make crop_seg_dict first, !THEN! embryo detection, !THEN! microssporidia/other channel. Other orders won't work as is.
    crop_seg_dict = list_out_images(inputfolder)
    pass_between_dict = predict_embryos(crop_seg_dict, embryo_model, embryos)
    final_dict = predict_microsporidia(pass_between_dict, microsporidia_model, microsporidia)
    csv_saver(embryos, microsporidia, save_csv, final_dict, inputfolder)
    # Save results as pickle for expanded datause
    filehandler =  open(os.pathjoin(inputfolder + "predictions.pickle"))
    pickle.dump(final_dict, filehandler)
    filehandler.close()
    print("Pickle saved.")
    return(final_dict)
#%%
res = detector(inputfolder = "input/folder/with/worm/predictions/and/cropped/dy96/", embryos = 1, microsporidia = 1)