# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:48:45 2024

@author: ebjam
"""
import cv2, os
import matplotlib.pyplot as plt
from scipy import ndimage

dy_wbw_path = "E:/toronto_microscopy/ixmc/Sep16_detergent_tests/selected_for_dy96_forest/"
clahe_path = dy_wbw_path + "clahe/"
no_clahe = dy_wbw_path + "no_clahe/"
os.makedirs(clahe_path, exist_ok=True)
os.makedirs(no_clahe, exist_ok=True)
dy_worm_by_worm = [q for q in os.listdir(dy_wbw_path) if q.endswith("TIF")]
log = []
for worm in dy_worm_by_worm:
    this_one = {}
    this_one['image'] = worm
    image = cv2.imread(os.path.join(dy_wbw_path, worm), cv2.IMREAD_GRAYSCALE)
    size = image.shape
    this_one['original_shape'] = size
    max_size = max(size)
    min_size = min(size)
    ar = max_size / min_size
    this_one['AR'] = ar
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(image) + 30
    if ar > 3:
        this_one['rotated'] = True
        rotate_45 = ndimage.rotate(final_img, 45, reshape = True, order = 0)
        raw_rotate_45 = ndimage.rotate(image, 45, reshape = True, order = 0)
        rotate_45_dims = [rotate_45.shape[0], rotate_45.shape[1]]
        this_one['rotation_dims'] = rotate_45_dims
        to_save = cv2.resize(rotate_45, (320, 320))
        no_clahe_img = cv2.resize(raw_rotate_45, (320, 320))
    else:
        this_one['rotated'] = False
        this_one['rotation_dims'] = (0,0)
        to_save = cv2.resize(final_img, (320, 320))
        no_clahe_img = cv2.resize(image, (320, 320))
    log.append(this_one)
    cv2.imwrite(clahe_path + "clahe_roto_resize" + worm, to_save)
    cv2.imwrite(no_clahe + "roto_resize" + worm, no_clahe_img)
        
#%%
import gzip, pickle
with gzip.open(clahe_path + 'aug3_n2_vs_ju1400_rotation_log.pkl.gz', 'wb') as f:
    pickle.dump(log, f)
print("Saved!")
