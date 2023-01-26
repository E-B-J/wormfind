# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:52:46 2023

@author: ebjam
"""
import cv2
import os
import numpy as np
import pandas as pd

def remove_trains(path, images_to_compare):
    images_to_compare_no_suffix = [q[:-13] for q in images_to_compare]
    images_no_filter = [w for w in os.listdir(path) if w.endswith(".png")]
    images_ = [e[:-15] for e in images_no_filter if e[:-15] in images_to_compare_no_suffix]
    missing_from_pred = [r for r in images_to_compare_no_suffix if r not in images_]
    missing_from_threshold = [t for t in images_ if t not in images_to_compare_no_suffix]
    union = images_
    return(len(images_to_compare), len(images_), missing_from_pred, missing_from_threshold, union)

def remove_trains2(path, images_to_compare):
    images_no_filter = [y for y in os.listdir(path) if y.endswith(".png")]
    images_ = [u[:-15] for u in images_no_filter if u[:-15] in images_to_compare]
    missing_from_pred = [i for i in images_to_compare if i not in images_]
    missing_from_threshold = [o for o in images_ if o not in images_to_compare]
    union = images_
    return(len(images_to_compare), len(images_), missing_from_pred, missing_from_threshold, union)

def get_image_names_from_union(images):
    threshold = [p + "threshold.png" for p in images]
    pred = [a + "pred_imsave.png" for a in images]
    return(threshold, pred)

def get_percent_thresh(images, path):
    values = []
    white_percent = []
    for image in images:
        img = cv2.imread(path + image)
        value = np.unique(img)
        values.append(value)
        number_of_white_pix = np.sum(img == 255)
        number_of_black_pix = np.sum(img == 0)
        total_px = number_of_black_pix + number_of_white_pix
        white_px_percent = number_of_white_pix/total_px
        white_percent.append(white_px_percent)
    return(white_percent, values)

def get_percent_pred(images, path):
    values = []
    white_percent = []
    for image in images:
        img = cv2.imread(path + image, cv2.IMREAD_GRAYSCALE)
        value = np.unique(img)
        values.append(value)
        number_of_white_pix = np.sum(img == 215)
        number_of_black_pix = np.sum(img == 30)
        total_px = number_of_black_pix + number_of_white_pix
        white_px_percent = number_of_white_pix/total_px
        white_percent.append(white_px_percent)
    return(white_percent, values)
#%%
eek = remove_trains("C:/Users/ebjam/Desktop/test/test/pred_out/", images)
eek2 = remove_trains2("C:/Users/ebjam/Downloads/pred_out_intensity_2-20230117T234138Z-001/pred_out_intensity_2/mod", eek[4])
thresholds, preds = get_image_names_from_union(eek2[4])
thresh = get_percent_thresh(thresholds, "C:/Users/ebjam/Desktop/test/test/thresholded/")
pred1 = get_percent_pred(preds, "C:/Users/ebjam/Desktop/test/test/pred_out/")
pred2 = get_percent_pred(preds, "C:/Users/ebjam/Downloads/pred_out_intensity_2-20230117T234138Z-001/pred_out_intensity_2/mod/")

    
#%%
file_to_dump = {"image": eek2[4], "thresh_white_percent": thresh[0], "pred1_white_percent": pred1[0],"pred2_white_percent":pred2[0]}

df = pd.DataFrame(file_to_dump)

df.to_csv("C:/Users/ebjam/Desktop/test/" + 'thresh_pred1_pred2_white_percent.csv') 