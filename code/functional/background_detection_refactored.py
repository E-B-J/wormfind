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
            
        FISH:
            For 'good' worms: Get percentage of worm body infected by microsporidia
            Specifically interrogates ribosome rRNA: presence of microsporidia 
            'fish_pixels_bg%' & 'fish_pixels_bg750%' - via simple thresholding based on channel background (+750AU)
            
    While running:
         Progress par to show estimated time until completion of the current plate.
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
from skimage.filters import sato
from skimage.util import img_as_ubyte
import joblib
import pandas as pd
from tqdm import tqdm

def load_worm_predictor(worm_clf_path):
    return joblib.load(worm_clf_path)
needed_for_worm_clf = ['circularity', 'perimeter', 'solidity', 'area', 'area_to_perimeter']

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

def make_mask_area(contour):
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    tran_cont = contour - [x, y]
    cv2.drawContours(mask, [tran_cont], -1, 255, thickness=-1)
    area = np.count_nonzero(mask)
    return(int(area))

def image_background_quant(dy96_img, fish_img, image_contours):
    '''
    This will take the background quantile for whatever image I'm looking at - I want to use this to threshold FISH & DY96 signal

    '''
    
    h, w = dy96_img.shape
    mask_for_circle = np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    # This should focus on center of well, and ignore the corners where there is only plastic, no signal - want a honest idea of what the background lookslike!
    radius = int((min(h, w) - 500) / 2)
    #Make mask
    cv2.circle(mask_for_circle, center, radius, 255, thickness = cv2.FILLED) # Dont want edge of well or dark field to bias measurement
    
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

def channel_paths_from_dapi(dapi_path):
    dy96_path = dapi_path.replace('/dapi/', '/dy96/')
    fish_path = dapi_path.replace('/dapi/', '/fish/')
    chans = [dapi_path, dy96_path, fish_path]
    return chans

def make_wbws(worm_by_worm, chans):
    if worm_by_worm == 1:
        wbws = [q + 'worm_by_worm' for q in chans]
        for wbw in wbws:
            os.makedirs(wbw, exist_ok=True)
        return wbws

def get_image_and_contours_from_dapi(dapi_path, image):
    # load image
    img = cv2.imread(os.path.join(dapi_path, image), -1)
    # Sato filter image
    vesselness = sato(img, black_ridges=False)
    # Process the vesselness image
    svesselness_ubyte = img_as_ubyte(vesselness)
    three_x = svesselness_ubyte * 3
    _, sv_thresh = cv2.threshold(three_x, 10, 255, cv2.THRESH_BINARY)
    # Well roi breaker here!! - stops well appreaing as only ROI
    top_left = (0,2000)  # Example position
    size = (400, 75)  # Example size (width, height)
    # Rectangle breaks well border - turns from an 'o' to a 'c'
    cv2.rectangle(sv_thresh, top_left, (top_left[0] + size[0], top_left[1] + size[1]), (0, 0, 0), -1)
    contours, _ = cv2.findContours(sv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours

def load_other_chans(dapi_image, chans):
    dy96_image = dapi_image.replace("w1_combo", "w2_combo")
    dy96_img = cv2.imread(os.path.join(chans[1], dy96_image), -1)
    fish_image = dapi_image.replace("w1_combo", "w3_combo")
    fish_img = cv2.imread(os.path.join(chans[2], fish_image), -1)
    return dy96_img, fish_img

def mask_and_crop(img, bbox, contour):
    mask = np.zeros(img.shape, dtype=np.uint16)
    cv2.drawContours(mask, [contour], -1, 65535, thickness=cv2.FILLED)
    masked_img = cv2.bitwise_and(img, mask)
    cropped_img = masked_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
    return cropped_img

def wbw_saver(contour, cropped_image, wbw_folder, lookup):
    # draw contour on image
    cv2.drawContours(cropped_image, [contour], -1, 65535, thickness=2)
    cv2.imwrite(wbw_folder + '/' + lookup + '.png', cropped_image)
    
def background_quant(cropped_img, bg99):
    return len(cropped_img[cropped_img > bg99]), len(cropped_img[cropped_img > bg99+750])

def worm_prediction(pred_meas, worm_clf):
    pred_meas.pop('contour')
    this_worm = pd.DataFrame(pred_meas) # I don't know why this makes multiple copies per worm, I'll just call the first instance in all cases
    to_predict = this_worm[needed_for_worm_clf].copy()
    worm_status = worm_clf.predict(to_predict)
    #print(worm_status[0])
    return worm_status[0]

def process_well(contours,
                 dapi_img, dy96_img, fish_img,
                 area_filters, 
                 image, 
                 worm_clf, 
                 base, 
                 worm_by_worm,
                 wbws):
    background_99_percentiles = image_background_quant(dy96_img, fish_img, contours)
    well_measurments=[]
    for q, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > area_filters[0]:
            if area < area_filters[1]:
                bbox = cv2.boundingRect(contour)
                # Get rid of well artifacts and odd merges
                #print(bbox[2:])
                if max(bbox[2:]) < 900:
                    lookup = image + "_worm_" + str(q)
                    d_lookup = lookup.replace("1_combo.TIF_", "2_combo_")
                    f_lookup = lookup.replace("1_combo.TIF_", "3_combo_")
                    perimeter = cv2.arcLength(contour, True)
                    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    dy_crop = mask_and_crop(dy96_img, bbox, contour)
                    fish_crop = mask_and_crop(fish_img, bbox, contour)
                    dy96_pixels_bg, dy96_pixels_bg750 = background_quant(dy_crop, background_99_percentiles['dy96_99'])
                    fish_pixels_bg, fish_pixels_bg750 = background_quant(fish_crop, background_99_percentiles['fish_99'])
                    adj_cont = transpose_contour(contour)
                    if worm_by_worm == 1:
                        wbw_saver(adj_cont, dy_crop, wbws[1], d_lookup)
                        wbw_saver(adj_cont, fish_crop, wbws[2], f_lookup)
                    mask_area = make_mask_area(contour)
                    measurements = {}
                    #print(contour)
                    measurements['contour'] = contour
                    #print(measurements['contour'])
                    measurements['image'] = image
                    measurements['lookup'] = lookup
                    measurements['area'] = area
                    measurements['perimeter'] = perimeter
                    measurements['area_to_perimeter'] = area / perimeter
                    measurements['circularity'] = circularity
                    measurements['solidity'] = solidity
                    measurements['bbox'] = bbox
                    measurements['fish_pixels_bg'] = fish_pixels_bg
                    measurements['fish_pixels_bg%'] = 100 * (fish_pixels_bg / mask_area)
                    measurements['fish_pixels_bg750'] = fish_pixels_bg750
                    measurements['fish_pixels_bg750%'] = 100 *(fish_pixels_bg750 / mask_area)
                    measurements['dy96_pixels_bg'] = dy96_pixels_bg
                    measurements['dy96_pixels_bg%'] = 100 * (dy96_pixels_bg / mask_area)
                    measurements['dy96_pixels_bg_750'] = dy96_pixels_bg750
                    measurements['dy96_pixels_bg_750%'] = 100 * (dy96_pixels_bg750 / mask_area)
                    measurements['mask_area'] = mask_area
                    measurements['fish_bg'] = background_99_percentiles['fish_99']
                    measurements['dy96_bg'] = background_99_percentiles['dy96_99']
                    meas = measurements.copy()
                    measurements['worm_prediction'] = worm_prediction(meas, worm_clf)
                    well_measurments.append(measurements)
    return well_measurments

def process_plate(plate, worm_by_worm):
    worm_clf = load_worm_predictor(r"C:\Users\ebjam\Documents\GitHub\wormfind\models\worm_detection_classifier\best_sato_Oct24_noZscore.joblib")
    chans = channel_paths_from_dapi(plate)
    wbws = make_wbws(worm_by_worm, chans)
    dapi_images = [q for q in os.listdir(chans[0]) if q.endswith("TIF")]
    plate_results = []
    for image in tqdm(dapi_images, desc=f"Processing plate: {plate}"):
        dapi_img, contours = get_image_and_contours_from_dapi(chans[0], image)
        dy96_img, fish_img = load_other_chans(image, chans)    
        inter = process_well(contours, dapi_img, dy96_img, fish_img, [2000, 100000], image, worm_clf, plate, worm_by_worm, wbws)
        plate_results.extend(inter)
    joblib.dump(plate_results, plate + "APR30_refactored_extend.joblib")
    
def worm_detection_and_analysis(folders):    
    for i, plate in enumerate(folders):
        if i == 0:
            print('Analyzing images - somewhat slow (~20s/image on Ed\'s laptop - may take a while before progress bar....progresses')
            print('Ed should really nut up and learn how to impliment multiprocessing!')
        process_plate(plate, 1)
# Main loop:a
folders = [           
        #"D:/toronto_microscopy/ixmc/OneDrive_1_5-20-2025_M20_R/M20r1r1-cb4037-4x-48hpi_Plate_2701/TimePoint_1/dapi/one_field/",
        #"D:/toronto_microscopy/ixmc/OneDrive_1_5-20-2025_M20_R/M20r1r2-cb4037-4x-48hpi_Plate_2702/TimePoint_1/dapi/one_field/",
        #"D:/toronto_microscopy/ixmc/M20r1r3-cb4037-4x-48hpi_Plate_2703/M20r1r3-cb4037-4x-48hpi_Plate_2703/TimePoint_1/dapi/one_field/",
        "D:/toronto_microscopy/ixmc/OneDrive_2_5-20-2025_M20_DRUGZ/M20Drug1_Plate_2704/TimePoint_1/dapi/one_field/",
        "D:/toronto_microscopy/ixmc/OneDrive_2_5-20-2025_M20_DRUGZ/M20Drug2_Plate_2705/TimePoint_1/dapi/one_field/",
        "D:/toronto_microscopy/ixmc/OneDrive_2_5-20-2025_M20_DRUGZ/M20Drug3_Plate_2706/TimePoint_1/dapi/one_field/",
        ]
        
worm_detection_and_analysis(folders)