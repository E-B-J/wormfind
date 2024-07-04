# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:03:52 2024

@author: ebjam
"""

# Get the size of worms - these are from manual annotations Jessie and Ed did.

import os, cv2
crop_path = "D:/Toronto_microscopy/OneDrive_1_11-28-2023_sds/2chan/SDS Wash U_Plate_1803/TimePoint_1/dy96/one_field/crops/"
images = [q for q in os.listdir(crop_path) if q.endswith('tif')]

xs = []
ys = []
for image in images:
    img = cv2.imread(os.path.join(crop_path, image))
    ys.append(img.shape[0])
    xs.append(img.shape[1])

maxi = max([max(xs), max(ys)])

buffer = 1.2 * maxi

print('Needed overlap = ' + str(buffer) + ' pixels')


#%%
# Crop input image into 4

n = 2
for i in range(n):
    print(i)
#%%
overlap = 792
onefield_path = 'D:/Toronto_microscopy/OneDrive_1_11-28-2023_sds/2chan/SDS Wash U_Plate_1803/TimePoint_1/dy96/one_field/test'
onefields = [q for q in os.listdir(onefield_path) if q.endswith('tif')]

for onefield in onefields:
    img = cv2.imread(os.path.join(onefield_path, onefield))
    h, w, c = img.shape
    # Get size of tile without overlap
    tile_h = h/n
    tile_w = w/n
    # Build tiles
    for i in range(n):
        for j in range(n):
            start_h = int(max(0, i*tile_h - overlap))
            end_h = int(min(h, (i+1)*tile_h + overlap))
            start_w = int(max(0, j*tile_w - overlap))
            end_w = int(min(w, (j+1)*tile_w + overlap))
            tile = img[start_h:end_h, start_w:end_w]
            # Predict on tile here!!!
            
            # Then mess around with the predictions
            
    