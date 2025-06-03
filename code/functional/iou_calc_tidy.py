# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 09:36:52 2024

@author: ebjam

iou calc portfolio level
Intended flow:
1 Load ai detections
2. image by image, load manual detections
3. make masks per image and compare
4. Save matches and non-matches



"""

import joblib, os, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Individual processes
def load_manual_data(dapi_one_dir):
    '''
    Parameters
    ----------
    dapi_one_dir : Directory with DAPI onefield ROI files
    Returns
    -------
    List of ROI csv files, named by image
    '''
    return([os.oath.join(dapi_one_dir, f) for f in os.listdir(dapi_one_dir) if f.endswith('csv')])

def load_ai_data(ai_joblib_path):
    return(pd.DataFrame(joblib.load(ai_joblib_path)))

# Function to show masks for debugging
# No return, but plots images
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
    
# Function to generate mask from contour for well size image - image size not variable!
# Returns mask from contour
def contour_to_mask_cv2(contour):
    mask = np.zeros((4096, 4096), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 65355, thickness=cv2.FILLED)
    return(mask)

# IOU from two binary masks
# Returns IoU as a 5dp float
def iou_two_masks(manual_mask, ai_mask):
    manual_mask = manual_mask.astype(bool)
    ai_mask = ai_mask.astype(bool)
    
    intersect = np.sum(manual_mask & ai_mask)
    union = np.sum(manual_mask | ai_mask)
    
    iou = round(intersect / union, 5)
    return(iou)

# Loop and logic
def image_loop_and_logic(csv_path, ai_detection_df):
    # Load data
    manual_data = pd.read_csv(csv_path)
    image_name = manual_data['Image Name'][0]
    relevant_ai_detections = ai_detection_df.loc[(ai_detection_df['image'] == image_name) &
                                                 ai_detection_df['worm_prediction'] == 'good'].copy()
    # Make a mask for each Ai detection
    ai_masks = []
    for w, ai_detection in relevant_ai_detections.iterrows():
        ai_masks.append({'image': image_name,
         'roi_name': ai_detection['lookup'],
         'mask': contour_to_mask_cv2(ai_detection['contour'])})
    # Initiate results
    unmatched_manual_detections = []
    unmatched_ai_detections = []
    matched_ai_detections = []
    # Compare manual detections against each AI detection
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
                    'IoU': iou})
        if len(roi_intersections) == 0:
            unmatched_manual_detections.append(detection.copy())
    # Log making/results
    matched_ai_detections = [r['ai_name'] for r in roi_intersections]
    all_ai_detections = list(relevant_ai_detections['lookup'].unique())
    unmatched_ai_detections = [t for t in all_ai_detections if t not in matched_ai_detections]
    log = {'image': image_name,
        'matched_detections': roi_intersections,
           'unmatched_manual_detections': unmatched_manual_detections,
           'unmatched_ai_detections': unmatched_ai_detections}
    return(log)
# Loop logic
def loop_through_images
