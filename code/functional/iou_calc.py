# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:29:03 2024

@author: ebjam

iou_calculation
"""
import joblib, os, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dapi_ones = "D:/toronto_microscopy/ixmc/RNai-nov14_Plate_2382/RNai-nov14_Plate_2382/TimePoint_1/dapi/one_field/"

csvs = [q for q in os.listdir(dapi_ones) if q.endswith("csv")]

ai_detections = pd.DataFrame(joblib.load(r"D:\toronto_microscopy\ixmc\RNai-nov14_Plate_2382\RNai-nov14_Plate_2382\TimePoint_1\dapi\one_field\Nov26_detections_dy96_and_fish3500.joblib"))

def show_masks(manual_mask, ai_mask, title="Masks Comparison"):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Manual Mask")
    plt.imshow(manual_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("AI Mask")
    plt.imshow(ai_mask, cmap='gray')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.show()

def contour_to_mask_cv2(contour):
    mask = np.zeros((4096, 4096), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 65355, thickness=cv2.FILLED)
    return(mask)

def iou_two_masks(manual_mask, ai_mask):
    manual_mask = manual_mask.astype(bool)
    ai_mask = ai_mask.astype(bool)
    
    intersect = np.sum(manual_mask & ai_mask)
    union = np.sum(manual_mask | ai_mask)
    
    iou = round(intersect / union, 5)
    return(iou)

unmatched_manual_annotation = []
unmatched_ai_detections = []
matched_detections = []    
intersections = []
for csv in csvs:
    manual_data = pd.read_csv(dapi_ones + csv)
    image = manual_data['Image Name'][0]
    rele_detect = ai_detections.loc[ai_detections['image'] == image].copy()
    rele_detect = rele_detect.loc[rele_detect['worm_prediction'] == "good"].copy()
    ai_masks = []
    for w, ai_detection in rele_detect.iterrows():
        ai_masks.append({'image': image,
         'roi_name': ai_detection['lookup'],
         'mask': contour_to_mask_cv2(ai_detection['contour'])})
    for e, detection in manual_data.iterrows():
        roi_intersections = []
        manual_contour = eval(detection['ROI Coordinates'])
        man_contour = np.array(manual_contour, dtype=np.int32).reshape((-1, 1, 2))
        man_mask = contour_to_mask_cv2(man_contour)
        for record in ai_masks:
            iou = iou_two_masks(man_mask, record['mask'])
            #print(iou)
            if iou > 0:
                #show_masks(man_mask, record['mask'])
                roi_intersections.append({
                    'image': detection['Image Name'],
                    'roi_name': detection['ROI Name'],
                    'ai_name': record['roi_name'],
                    'IoU': iou,
                    'fiji_contour': detection['ROI Coordinates'],
                    'ai_contour': rele_detect.loc[rele_detect['lookup'] == record['roi_name']]['contour']})
                    # Wanted contours for ease of analysis
        if len(roi_intersections) == 0:
            unmatched_manual_annotation.append(detection.copy())
        else:
            for arbitrary in roi_intersections:
                intersections.append(arbitrary)
    matched_ai_detections = [r['ai_name'] for r in roi_intersections]
    all_ai_detections = list(rele_detect['lookup'].unique())
    unmatched_ai_detections_list = [t for t in all_ai_detections if t not in matched_ai_detections]
    for y in unmatched_ai_detections_list:
        unmatched_ai_detections.append(y)
        

plate_res = {'unmatched_manual': unmatched_manual_annotation,
             'intersections': intersections,
             'unmatched_ais': unmatched_ai_detections}

joblib.dump(plate_res, dapi_ones + 'iou_results.joblib')