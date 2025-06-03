# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:58:37 2024

@author: old bojangles

Before running: Stitch your well views together! - huh, maybe I shuold make that a  nice little function and run it first here...
        TODO: Functionalize per channel, and the function to connect channels, then run first!

This code will:
    Take well views and:
        DAPI:
            Run a sato filter to seperate objects from background
            Predict if each of those objects is a worm using simple morphological features
            Pass locations of 'good' detections to other channels
        DY96:
            For 'good' worms: Get percentage of worm body infected by microsporidia
            Specifically interrogates spore formation: completeion of microsporidia lifecycle
            Two metrics: 'dy96_percent' - via simple thresholding @ 10,000/65,355 (might be too low!), and 'dy96_pred_percent' - via a spore segmentation network
            
            Elso embryo detection - 320x320 embryo detection colab book
        FISH:
            For 'good' worms: Get percentage of worm body infected by microsporidia
            Specifically interrogates ribosome rRNA: presence of microsporidia 
            Two metrics: 'fish_percent' - via simple thresholding, and also bg%s via looking at background intensity
            
    While running:
         Prints the name of the current image - you can use this to estimate how long you have left.
    Output:
        Object called: 'plate_results' - a list of dictionaries, with each detected worm being it's own dictionary. 
        You can turn this into a dataframe with pd.DataFrame(plate_results),
        and then save it as a csv or whatever your prefered flavour of datafile is.


I frequently don't understand all of this code at once, but it seems to work!
Make edits at your own risk!

Ed
"""

import os
import cv2
import numpy as np
from skimage import feature
from skimage.filters import sato
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from scipy import ndimage
import joblib
from functools import partial
import pandas as pd

worm_clf_path = r"C:\Users\ebjam\Documents\GitHub\wormfind\models\worm_detection_classifier\best_sato_Oct24_noZscore.joblib"
worm_clf = joblib.load(worm_clf_path)
needed_for_worm_clf = ['circularity', 'perimeter', 'solidity', 'area', 'area_to_perimeter']
#dy96_clf_path = "E:/toronto_microscopy/ixmc/sep_30_success/training_dy96_segmentation/noclahe/best_RF_DY96_acc=_Oct24.joblib"
#dy96_clf = joblib.load(dy96_clf_path)
#fish_clf_path = "E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/fish/best_rf_aug27.joblib"
#fish_clf = joblib.load(fish_clf_path)
sig_min = 1
sig_max = 16
features_func = partial(
    feature.multiscale_basic_features,
    intensity = True,
    edges = True,
    texture = True,
    sigma_min = sig_min,
    sigma_max = sig_max,
    )

clahe = cv2.createCLAHE(clipLimit=5)

kernel = np.ones((3, 3), np.uint8)

def transpose_contour(contour):
    transposed_contour = []
    bbox = cv2.boundingRect(contour)
    for point in contour:
        for item in point:
            trans_x = item[0] - bbox[0]
            trans_y = item[1] - bbox[1]
            transposed_contour.append([[trans_x, trans_y]])
    np_transposed_contour = np.array(transposed_contour)
    return(np_transposed_contour)

def unrotate(fish_prediction_img, clahe_cmf, rotate_45_dims):
    ori_x, ori_y = clahe_cmf.shape[0:2]
    rotate_45 = cv2.resize(fish_prediction_img, rotate_45_dims, interpolation = cv2.INTER_NEAREST)
    rotate_back = ndimage.rotate(rotate_45, 315, reshape = True, order = 0)
    re_roto_x, re_roto_y = rotate_back.shape
    extra_x = re_roto_x - ori_x
    extra_y = re_roto_y - ori_y
    x_padding = int(extra_x/2)
    y_padding = int(extra_y/2)
    x_start = int(x_padding)
    x_end = int(re_roto_x) - int(x_padding)
    y_start = int(y_padding)
    y_end = int(re_roto_y) - int(y_padding)
    # Crop is off!!!
    cropped = rotate_back[x_start:x_end, y_start:y_end]
    orig_size_fish_prediction = cv2.resize(cropped, (ori_y, ori_x), interpolation = cv2.INTER_NEAREST)
    return(orig_size_fish_prediction)

folders = [r"D:/toronto_microscopy/ixmc/RNai-nov14_Plate_2382/RNai-nov14_Plate_2382/TimePoint_1/dapi/one_field/",
           r"D:/toronto_microscopy/ixmc/RNai-p2-nov14_Plate_2383/RNai-p2-nov14_Plate_2383/TimePoint_1/dapi/one_field/",
           r"D:/toronto_microscopy/ixmc/rnAI-rep3-dec6_Plate_2441/rnAI-rep3-dec6_Plate_2441/TimePoint_1/dapi/one_field/"]

def image_background_quant(dy96_img, fish_img, image_contours):
    '''
    This will take the background quantile for whatever image I'm looking at - I want to use this to threshold FISH signal

    '''
    #Make mask
    center = (2048, 2048)
    radius = int((4096 - 500) / 2) # This should focus on center of well, and ignore the corners where there is only plastic, no signal - want a honest idea of what the background lookslike!
    mask_for_circle = np.zeros((4096, 4096), dtype=np.uint8) # Size of well image - could functionalize!
    cv2.circle(mask_for_circle, center, radius, 255, thickness = cv2.FILLED) # Dont want edge of well or dark field to bias measurement
    mask = np.zeros((4096, 4096), dtype=np.uint8)
    cv2.drawContours(mask, image_contours, -1, 255, thickness=cv2.FILLED) # Add all detections ot mask
    mask_outside = cv2.bitwise_not(mask) # Look OUTSIDE detections
    background = cv2.bitwise_and(mask_outside, mask_for_circle)
    # DY96 background values
    dy96_background = cv2.bitwise_and(dy96_img, dy96_img, mask = background)
    dy96_bg = np.ma.masked_equal(dy96_background, 0)
    dy96_bgl = dy96_bg.compressed().tolist()
    
    # FISH background values
    fish_background = cv2.bitwise_and(fish_img, fish_img, mask = background)
    fish_bg = np.ma.masked_equal(fish_background, 0)
    fish_bgl = fish_bg.compressed().tolist()
    
    dy96_99 = np.percentile(dy96_bgl, 99)
    fish_99 = np.percentile(fish_bgl, 99)
    return {'dy96_99':dy96_99, 'fish_99':fish_99}
    
# Make this individual functions, and maybe allow fish and DY 96 to be optioned for DY96 only stuff?
for plate in folders:
    ju1400_1_DAPI = plate
    ju1400_1_DY96 = ju1400_1_DAPI.replace("/dapi/", "/dy96/")
    ju1400_1_FISH = ju1400_1_DAPI.replace("/dapi/", "/fish/")
    
    chans = [ju1400_1_DAPI, ju1400_1_DY96, ju1400_1_FISH]
    
    dapi_wbw = ju1400_1_DAPI + "worm_by_worm/"
    dy96_wbw = ju1400_1_DY96 + "worm_by_worm/"
    fish_wbw = ju1400_1_FISH + "worm_by_worm/"
    
    wbws = [dapi_wbw, dy96_wbw, fish_wbw]
    
    for wbw in wbws:
        os.makedirs(wbw, exist_ok=True)
    # Add worm by worms here
    
    ju_1_dapi_images = [q for q in os.listdir(ju1400_1_DAPI) if q.endswith("TIF")]
    plate_results = []
    
    for image in ju_1_dapi_images:
        print(image)
        img = cv2.imread(os.path.join(ju1400_1_DAPI, image), -1)  # Load the image
        vesselness = sato(img, black_ridges=False)  # Apply Sato filter
        dy96_image = image.replace("w1_combo", "w2_combo")
        dy96_img = cv2.imread(os.path.join(ju1400_1_DY96, dy96_image), -1)
        fish_image = image.replace("w1_combo", "w3_combo")
        fish_img = cv2.imread(os.path.join(ju1400_1_FISH, fish_image), -1)
        # Process the vesselness image
        svesselness_ubyte = img_as_ubyte(vesselness)
        three_x = svesselness_ubyte * 3
        _, sv_thresh = cv2.threshold(three_x, 10, 255, cv2.THRESH_BINARY)
    
        # Find contours
        contours, _ = cv2.findContours(sv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        simplified_contours = []
        epsilon = 0.001 * cv2.arcLength(contours[0], True)  # Adjust epsilon for smoothing
        namer = 0
        this_well = []
        
        # Background quantification!!!
        
        background_99_percentiles = image_background_quant(dy96_img, fish_img, contours)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                if area < 100000:
                    
                    lookup = image + "_worm_" + str(namer)
                    measurements = {}
                    measurements['image'] = image
                    measurements['lookup'] = lookup
                    measurements['area'] = area
                    perimeter = cv2.arcLength(contour, True)
                    measurements['perimeter'] = perimeter
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                    measurements['circularity'] = circularity
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    measurements['solidity'] = solidity
                    bbox = cv2.boundingRect(contour)
                    measurements['bbox'] = bbox
                    x, y, w, h = bbox
                    mask = np.zeros(img.shape, dtype=np.uint16)
                    cv2.drawContours(mask, [contour], -1, 65535, thickness=cv2.FILLED)  # Fill the contour
                    # Extract the pixel values within the contour
                    masked_image = cv2.bitwise_and(img, mask)
                    masked_dimage = cv2.bitwise_and(dy96_img, mask)
                    masked_fimage = cv2.bitwise_and(fish_img, mask)
                    cropped_dapi = img[y:y+h, x:x+w]
                    cropped_dy96 = dy96_img[y:y+h, x:x+w]
                    cropped_fish = fish_img[y:y+h, x:x+w]
                    
                    # Do FISH and DY96 prediction here!
                    #cv2.imwrite(dapi_wbw + lookup + ".TIF", cropped_dapi)
                    d_lookup = lookup.replace("1_combo.TIF_", "2_combo_")
                    #cv2.imwrite(dy96_wbw + d_lookup + ".TIF", cropped_dy96)
                    f_lookup = lookup.replace("1_combo.TIF_", "3_combo_")
                    #cv2.imwrite(fish_wbw + f_lookup + ".TIF", cropped_fish)
                    crop_mask_dy96 = masked_dimage[y:y+h, x:x+w]
                    num_DY96_pixels_10k = len(crop_mask_dy96[crop_mask_dy96 > 10000])
                    num_DY96_pixels_5k = len(crop_mask_dy96[crop_mask_dy96 > 5000])
                    crop_mask_fish = masked_fimage[y:y+h, x:x+w]
                    num_fish_pixels_10k = len(crop_mask_fish[crop_mask_fish > 10000])
                    num_fish_pixels_8k = len(crop_mask_fish[crop_mask_fish > 8000])
                    num_fish_pixels_5k = len(crop_mask_fish[crop_mask_fish > 5000])
                    num_fish_pixels_3p5k = len(crop_mask_fish[crop_mask_fish > 3500])
                    num_fish_pixels_2k = len(crop_mask_fish[crop_mask_fish > 2000])
                    num_fish_pixels_1k = len(crop_mask_fish[crop_mask_fish > 1000])
                    num_fish_pixels_bg = len(crop_mask_fish[crop_mask_fish > background_99_percentiles['fish_99']]) # Based on any non worm signal
                    num_fish_pixels_bg_plus500 = len(crop_mask_fish[crop_mask_fish > (background_99_percentiles['fish_99']+500)]) # Exclude low worm signal
                    
                    measurements['contour'] = contour
                    measurements['dy96_pixels_10k'] = num_DY96_pixels_10k
                    measurements['dy96_pixels_5k'] = num_DY96_pixels_5k
                    measurements['dy96_percent_10k'] = 100 * (num_DY96_pixels_10k / area)
                    measurements['fish_pixels_10k'] = num_fish_pixels_10k
                    measurements['fish_pixels_3p5k'] = num_fish_pixels_3p5k
                    measurements['fish_pixels_2k'] = num_fish_pixels_2k
                    measurements['fish_pixels_1k'] = num_fish_pixels_1k
                    measurements['fish_pixels_bg'] = num_fish_pixels_bg
                    measurements['fish_pixels_bg500'] = num_fish_pixels_bg_plus500
                    measurements['fish_bg'] = background_99_percentiles['fish_99']
                    
                    measurements['fish_pixels_5k'] = num_fish_pixels_5k
                    measurements['fish_pixels_8k'] = num_fish_pixels_8k
                    measurements['fish_percent_8k'] = 100 * (num_fish_pixels_8k / area)
                    measurements['area_to_perimeter'] = area / perimeter
                    
                    
                    
                    # Now make fish mask...
                    meront_dy  = crop_mask_dy96 * (crop_mask_fish > 2000)
                    meront_num = len(meront_dy[meront_dy > 5000])
                    measurements['meront_num'] = meront_num
                    measurements['meront_%'] = meront_num / area
                    
                    meront_dy_bg = crop_mask_dy96 * (crop_mask_fish > background_99_percentiles['fish_99'])
                    meront_num_bg = len(meront_dy_bg[meront_dy > 5000])
                    measurements['meront_num_bg'] = meront_num_bg
                    measurements['meront_%_bg'] = meront_num_bg / area
                    
                    meront_dy_bg500 = crop_mask_dy96 * (crop_mask_fish > (background_99_percentiles['fish_99']+500))
                    meront_num_bg500 = len(meront_dy_bg500[meront_dy > 5000])
                    measurements['meront_num_bg500'] = meront_num_bg500
                    measurements['meront_%_bg500'] = meront_num_bg500 / area
                    # Lets also use the BG fish to take a look at meront percent - this will be meront bg %
                    
                    
                    
                    
                    
                    #Need to do the worm classification here!! - if it makes sense, then 
                    approx = cv2.approxPolyDP(contour, epsilon, True)  # Simplify the contour
                    measurements['approximate_contour'] = approx
                    pred_meas = measurements.copy()
                    pred_meas.pop('contour')
                    pred_meas.pop('approximate_contour')
                    this_worm = pd.DataFrame(pred_meas) # I don't know why this makes multiple copies per worm, I'll just call the first instance in all cases
                    to_predict = this_worm[needed_for_worm_clf].copy()
                    worm_status = worm_clf.predict(to_predict)
                    measurements['worm_prediction'] = worm_status[0] # Here is the first instance calling
                    if worm_status[0] == 'good':
                        cv2.imwrite(dy96_wbw + d_lookup + ".TIF", cropped_dy96)
                        '''
                        make_ar = [w, h]
                        clahe_fworm = clahe.apply(cropped_fish) + 30
                        aspect_ratio = max(make_ar) / min(make_ar)
                        measurements['aspect_ratio'] = aspect_ratio
                        if aspect_ratio > 3:
                            measurements['rotated'] = True
                            rotate_45_fish = ndimage.rotate(clahe_fworm, 45, reshape = True, order = 0)
                            rotate_45_dim = rotate_45_fish.shape[0]
                            measurements['rotate_45_dim'] = rotate_45_dim
                            resized_cfworm = cv2.resize(rotate_45_fish, (320, 320))
                            rotate_45_dy96 = ndimage.rotate(cropped_dy96, 45, reshape = True, order = 0)
                            resized_dworm = cv2.resize(rotate_45_dy96, (320, 320))
                        else:
                            measurements['rotated'] = False
                            measurements['rotate_45_dim'] = None
                            resized_cfworm = cv2.resize(clahe_fworm, (320,320))
                            resized_dworm = cv2.resize(cropped_dy96, (320,320))
                            
                        # Great, now extract waaaayyyyyyyy too many features:
                        
                        features = features_func(resized_cfworm)
                        X = features.reshape(-1, features.shape[-1])
                        fish_prediction = fish_clf.predict(X)
                        fish_prediction_img = fish_prediction.reshape((320,320))
                        
                        dy_features = features_func(resized_dworm)
                        dyX = features.reshape(-1, dy_features.shape[-1])
                        dy_prediction = dy96_clf.predict(dyX)
                        dy_pred_img = dy_prediction.reshape((320,320))
                        # Make bbox adjusted contour here!
                        transposed_contour = transpose_contour(contour)
                        #now I need to make it into a binary mask
                        mask_for_preds = np.zeros(cropped_dapi.shape, dtype=np.uint8)
                        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                        resized_mask = cv2.resize(mask_for_preds, (320, 320))
                        # Now, I need to use that mask to mask out the predictions
                        #masked_fish_pred = cv2.bitwise_and(fish_prediction_img, resized_mask)
                        masked_dy96_pred = cv2.bitwise_and(dy_pred_img, resized_mask)
                        #num_fish_pixels = np.sum(masked_fish_pred == 253)
                        num_dy96_pixels = np.sum(masked_dy96_pred == 4) #!!!
                        #measurements['#fishpred'] = num_fish_pixels
                        #measurements['fish_pred_percent'] = 100* (num_fish_pixels / area)
                        measurements['#dy96pred'] = num_dy96_pixels
                        measurements['dy96_pred_percent'] = 100 * (num_dy96_pixels / area)
                   ''' 
                    simplified_contours.append(approx)
                    this_well.append(measurements)
                    plate_results.append(measurements)
                    namer += 1
    
    joblib.dump(plate_results, ju1400_1_DAPI + "FEB27_detections_dy96_and_fisg_bg_quant.joblib")
    
#%%   
joblib.dump(plate_results, ju1400_1_DAPI + "Nov26_detections_dy96_and_fish3500.joblib") 
#%%
print(type(img.shape))
#%%
t_cont = transpose_contour(contour)
boundr = cv2.boundingRect(t_cont)
print(boundr)
#%%
#Need to transpose contour - I had that worked out earlier!
stand_in_mask = np.zeros((boundr[2], boundr[3]), dtype=np.uint16)
cv2.drawContours(stand_in_mask, [t_cont], -1, 65535, thickness=cv2.FILLED)
plt.imshow(stand_in_mask)
#%%
# Masks is making weird! NEed tio remake the mask to crop out only rele signal here!

plt.imshow(fish_prediction_img)
#%%
plt.imshow(resized_mask)