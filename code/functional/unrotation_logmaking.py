# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:40:38 2024

@author: ebjam

unrotation based on logmaking rotations
"""

import pickle, gzip
import cv2, os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage

def unrotate(fish_prediction_img, o_dim, rotate_45_dims):
    ori_x, ori_y = o_dim
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


root_path = "E:/toronto_microscopy/ixmc/sep_30_success/JU1400_reps/dy96/one_field/worm_by_worm/noclahe/"
resizes = root_path + "o_size_preds/"
os.makedirs(resizes, exist_ok=True)
rotation_log_path = root_path + "rotation_log.pkl.gz"

with gzip.open(rotation_log_path, 'rb') as f:
    log = pickle.load(f)
#%%

for item in log:
    image = item['image']
    rotated = item['rotated']
    o_dim = item['original_shape']
    r_dim = item['rotation_dims']
    
    prediction_image_handle = "roto_resize" + image.replace(".TIF", "_Simple Segmentation.png")
    pred_path = root_path + prediction_image_handle
    o_path = root_path.replace("noclahe/", "")
    o_img = cv2.imread(os.path.join(o_path, image))
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if rotated == False:
        print('no rotation')
        pred_resize = cv2.resize(pred_img, (o_dim[1], o_dim[0]), interpolation = cv2.INTER_NEAREST)
    elif rotated == True:
        pred_resize = unrotate(pred_img, o_dim, r_dim)
    
    o_img_rgb = cv2.cvtColor(o_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))  # Set the figure size
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.imshow(o_img_rgb)
    plt.title("Original Image")
    plt.axis('off')  # Turn off axis
    
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.imshow(pred_resize, cmap='viridis')  # Display one-hot encoded image
    plt.title("Predicted Image (One-Hot)")
    plt.axis('off')  # Turn off axis
    
    plt.tight_layout()  # Adjust spacing between plots
    plt.show()
    cv2.imwrite(os.path.join(resizes, image.replace(".TIF", "_pred_o_dim.png")), pred_resize)