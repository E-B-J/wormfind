# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:48:45 2024

@author: ebjam
"""
import cv2, os
from scipy import ndimage
import gzip, pickle

dy_wbw_path = "E:/toronto_microscopy/ixmc/OneDrive_1_11-22-2024/alb-3of3-nov14_Plate_2384/TimePoint_1/dy96/one_field/worm_by_worm/"
#clahe_path = dy_wbw_path + "clahe/"
noclahe_path = dy_wbw_path + "noclahe/"
#os.makedirs(clahe_path, exist_ok=True)
os.makedirs(noclahe_path, exist_ok=True)
dy_worm_by_worm = [q for q in os.listdir(dy_wbw_path) if q.endswith("TIF")]
log = []
for worm in dy_worm_by_worm:
    this_one = {}
    this_one['image'] = worm
    img = cv2.imread(os.path.join(dy_wbw_path, worm), -1)
    size = img.shape
    this_one['original_shape'] = size
    max_size = max(size)
    min_size = min(size)
    ar = max_size / min_size
    this_one['AR'] = ar
 #   clahe = cv2.createCLAHE(clipLimit=5)
    if ar > 3:
        this_one['rotated'] = True
        rotate_45 = ndimage.rotate(img, 45, reshape = True, order = 0)
        rotate_45_dims = [rotate_45.shape[0], rotate_45.shape[1]]
        this_one['rotation_dims'] = rotate_45_dims
        rot_img = cv2.resize(rotate_45, (320, 320))
    else:
        this_one['rotated'] = False
        this_one['rotation_dims'] = (0,0)
        rot_img = cv2.resize(img, (320, 320))
  #  final_img = clahe.apply(rot_img) + 30
    log.append(this_one)
    #cv2.imwrite(clahe_path + "clahe_roto_resize" + worm, final_img)
    cv2.imwrite(noclahe_path + "roto_resize" + worm, rot_img)

with gzip.open(noclahe_path + 'rotation_log.pkl.gz', 'wb') as f:
    pickle.dump(log, f)
print("Saved!")
