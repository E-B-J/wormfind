# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:18:27 2023

@author: ebjam
"""

import pickle

with open('C:/Users/ebjam/Downloads/predictions_GPU (1).pickle', 'rb') as f:
    # Load the object from the file
    my_object = pickle.load(f)
    
#%%
gravid = []
for image in my_object:
    for q in range(0, len(image['single_worms'])-1):
        worm = image['single_worms'][q]
        if worm['#_internal_embryos'] > 0:
            gravid.append(worm)

from shapely.geometry import Polygon
import numpy as np
from functools import reduce
import rasterio.features
from rasterio.transform import from_bounds

def embryo_masker(gravid_dict):
    for worm in gravid_dict:
        xmin, ymin, xmax, ymax = worm['bbox']
        width = xmax - xmin
        height = ymax - ymin
        embryomasks = worm['internal_embryo_masks']
        for mask in embryomasks:
            if mask[0] != mask[-1]:
                mask.append[mask[0]]
            else:
                break
        polygons = [Polygon(p) for p in embryomasks]
        # Turn polygons into a mask of the union of polygons.
        boolean_masks = []
        for polygon in polygons:
            # Define the output shape of the mask
            mask_shape = (height, width)
            
            # Create a 2D grid of coordinates
            x, y = np.meshgrid(np.arange(mask_shape[1]), np.arange(mask_shape[0]))
            
            # Convert the coordinates to the CRS of the polygon (if necessary)
            # ...
            
            # Check if the polygon is valid (not self-intersecting, etc.)
    
            
            # Create the mask from the polygon
            transform = from_bounds(0, 0, width, height, width, height)
            mask = rasterio.features.geometry_mask([polygon], out_shape=mask_shape, transform=transform, invert=True)
            boolean_masks.append(mask)
            
        stack_mask = reduce(lambda x, y: x | y, boolean_masks)
        binary_arr = stack_mask.astype(int) * 255
        arr_flipped = np.flipud(binary_arr)
        worm['embryo_combo_mask'] = arr_flipped
    return(gravid_dict)

#%%
g2 = embryo_masker(gravid)
