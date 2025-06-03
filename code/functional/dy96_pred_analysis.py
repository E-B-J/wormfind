# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:00:09 2024

@author: ebjam

Processing DY96 predictions

Will need to load all detections,
then go worm by worm:
    *Load image
    *Mask image
    *Measure image - 4 is spore

"""

import joblib, cv2
import numpy as np

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

kernel = np.ones((3, 3), np.uint8)

plates = ["D:/toronto_microscopy/ixmc/alb-dex-dec3-1_Plate_2421/alb-dex-dec3-1_Plate_2421/TimePoint_1/dapi/one_field/",
         "D:/toronto_microscopy/ixmc/alb-dex-dec3-2_Plate_2422/alb-dex-dec3-2_Plate_2422/TimePoint_1/dapi/one_field/",
         "D:/toronto_microscopy/ixmc/alb-dex-dec3-3_Plate_2423/alb-dex-dec3-3_Plate_2423/TimePoint_1/dapi/one_field/"]

for plate in plates:
    root = plate
    pred_path = root.replace('dapi', 'dy96') + "worm_by_worm/noclahe/o_size_preds/"
    detections = joblib.load(root + "Nov26_detections_dy96_and_fish3500.joblib") # a list right now!
    
    for detection in detections:
        if detection['worm_prediction'] == 'good':
            lookup = detection['lookup']
            t_contour = transpose_contour(detection['contour'])
            bbox = cv2.boundingRect(t_contour) #x, y, w, h
            
            mask = np.zeros((bbox[3], bbox[2]), dtype=np.uint8)
            cv2.drawContours(mask, [t_contour], contourIdx=-1, color=255, thickness=-1)
            detection['mask_area'] = len(mask[mask == 255])
            
            pred_lookup = lookup.replace("w1_combo.TIF", "w2_combo") + "_pred_o_dim.png"
            pred_img = cv2.imread(pred_path + pred_lookup, cv2.IMREAD_GRAYSCALE)
            spore_pred_img = np.where(pred_img == 4, 255, 0)
            spi_8 = spore_pred_img.astype('uint8')
            
            masked_spi_8 = cv2.bitwise_and(spi_8, spi_8, mask=mask)
            ep0_px = len(masked_spi_8[masked_spi_8 > 0])
            ero_pred_1 = cv2.erode(spi_8, kernel, iterations=1)
            ep1_px = len(ero_pred_1[ero_pred_1 > 0])
            ero_pred_2 = cv2.erode(spi_8, kernel, iterations=2)
            ep2_px = len(ero_pred_2[ero_pred_2 > 0])
            ero_pred_3 = cv2.erode(spi_8, kernel, iterations=3)
            ep3_px = len(ero_pred_3[ero_pred_3 > 0])
            
            detection['dy96_ep0_px'] = ep0_px
            detection['dy96_ep0_%'] = 100 * (ep0_px / detection['mask_area'])
            detection['dy96_ep1_px'] = ep1_px
            detection['dy96_ep1_%'] = 100 * (ep1_px / detection['mask_area'])
            detection['dy96_ep2_px'] = ep2_px
            detection['dy96_ep2_%'] = 100 * (ep2_px / detection['mask_area'])
            detection['dy96_ep3_px'] = ep3_px
            detection['dy96_ep3_%'] = 100 * (ep3_px / detection['mask_area'])
            print('tick')
    joblib.dump(detections, root.replace('dapi', 'dy96') + "Nov26_detections_with_Nov26_dy96_ilastik.joblib")

#%%

print(len(mask[mask == 0]))