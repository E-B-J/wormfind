# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:25:07 2024

@author: ebjam
"""
import pickle, cv2, os
import pandas as pd
import numpy as np

dy_path = "E:/toronto_microscopy/ixmc/Oct_8_ALB & N2vsAWR73/Oct 08 ALB RT_Plate_2301/TimePoint_1/dy96/one_field/"
embryo_wbw = os.path.join(dy_path, "embryo_wbw/")
os.makedirs(embryo_wbw, exist_ok=True)
embryo_boost_wbw = os.path.join(dy_path, "embryo_boost_wbw/")
os.makedirs(embryo_boost_wbw, exist_ok=True)
with open(r'E:\toronto_microscopy\ixmc\Oct_8_ALB & N2vsAWR73\Oct 08 ALB RT_Plate_2301\TimePoint_1\dapi\one_field\n2_alb_rep1_dapi_sato_unfiltered_oct21.pkl', 'rb') as f:
    data = pickle.load(f)
im_path = ""
wbw_centered = os.path.join(im_path + "wbw_centered/")
if not os.path.exists(wbw_centered):
    os.makedirs(wbw_centered)
df = pd.DataFrame(data)
unique_images = df['image'].unique()
for image in unique_images:
    rele_detect = df[df['image'] == image]
    dy_handle = image.replace("w1", "w2")
    dy_img = cv2.imread(os.path.join(dy_path + dy_handle), -1)
    for i, detect in rele_detect.iterrows():
        bbox = detect['bbox']
        lookup = detect['lookup'] + ".png"
        centroid = (bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2))
        centered_bbox = (centroid[0] - 160, centroid[1] - 160, 320, 320)
        cropped_img = dy_img[int(centered_bbox[1]):int(centered_bbox[1] + centered_bbox[3]), int(centered_bbox[0]):int(centered_bbox[0] + centered_bbox[2])]
        boost = np.copy(cropped_img)
        mask = cropped_img < 15000
        boost[mask]*=5
        boost[~mask] = 65535 
        boost = np.clip(boost, 0, 65535)
        cv2.imwrite(os.path.join(embryo_wbw + lookup), cropped_img)
        cv2.imwrite(os.path.join(embryo_boost_wbw + lookup), boost)
        
    