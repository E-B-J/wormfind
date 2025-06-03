# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 08:54:57 2025

@author: ebjam

image stitching functionalized
"""

import os
import cv2
from tqdm import tqdm
import csv
#Make folder for each channel
def prepare_channel_dirs(base_path):
    '''
    Parameters
    ----------
    base_path : raw path downloaded from ixmc

    Returns
    -------
    chan_paths :list of newly created channel paths within base_path. 

    '''
    # Change channels as necessary!
    channels = ['dapi', 'dy96', 'fish']
    chan_paths = []
    for chan in channels:
        path = base_path + '/' + chan
        os.makedirs(path + '/' + 'one_field', exist_ok=True)
        chan_paths.append(path)
    return chan_paths

#Get all TIF files in folder
def get_image_lists(plate_path):
    '''
    Parameters
    ----------
    plate_path : raw path downloaded from ixmc
    
    Returns
    ----------
    three lists of images split by wavelength
    '''
    all_imgs = [f for f in os.listdir(plate_path) if f.endswith('TIF')]
    #Split via wavelength
    get_imgs = lambda w: [f for f in all_imgs if f[-5] == str(w)]
    # Return three lists split by wavelength
    return get_imgs(1), get_imgs(2), get_imgs(3)

def read_and_stitch_tiles(w_images, base_path, chan_path, channel_label, errors):
    '''
    

    Parameters
    ----------
    w_images : all images of a plate that are a specified wavelength
    base_path : plate base path
    chan_path : wavelength folder
    channel_label : label for the channel (w1, w2, w3)
    errors : lsit to update with all images which fail out

    Returns
    -------
    errors : updated error list

    '''
    
    # s1 = top left
    s1_imgs = [img for img in w_images if img[-8] == '1']
    #prog bar for plate
    for top_left in tqdm(s1_imgs, desc = f"Stitching {channel_label}", unit=' wells'):
        # Construct neighboring tile names
        tr = top_left[:-8] + '2' + top_left[-7:]
        bl = top_left[:-8] + '3' + top_left[-7:]
        br = top_left[:-8] + '4' + top_left[-7:]
        
        # Read images
        s1 = cv2.imread(base_path + '/' + top_left, -1)
        s2 = cv2.imread(base_path + '/' + tr, -1)
        s3 = cv2.imread(base_path + '/' + bl, -1)
        s4 = cv2.imread(base_path + '/' + br, -1)
        
        if any(img is None for img in (s1, s2, s3, s4)):
            print("Warning: Missing tile for " + top_left)
            errors.append(top_left)
            continue
        
        # Stitch
        top = cv2.hconcat([s1, s2])
        bottom = cv2.hconcat([s3, s4])
        stitched = cv2.vconcat([top, bottom])
        
        outname = top_left[:-10] + f'{channel_label}_combo.TIF'
        outpath = chan_path + '/one_field/' + outname
        cv2.imwrite(outpath, stitched)
    return errors

def process_plate(base_path, errors):
    '''
    Parameters
    ----------
    base_path :raw path downloaded from ixmc
    errors : updated error list

    Returns
    -------
    updated error list

    '''
    dapi_path, dy96_path, fish_path = prepare_channel_dirs(base_path)
    w1_imgs, w2_imgs, w3_imgs = get_image_lists(base_path)
    errors = read_and_stitch_tiles(w1_imgs, base_path, dapi_path, 'w1', errors)
    errors = read_and_stitch_tiles(w2_imgs, base_path, dy96_path, 'w2', errors)
    errors = read_and_stitch_tiles(w3_imgs, base_path, fish_path, 'w3', errors)
    return errors

def stitch_paths(base_paths):
    all_errors = {}
    print('Stitching images')
    for path in base_paths:
        errors = []
        print(f'Processing: {path}')
        process_plate(path, errors)
        all_errors[path] =  errors
    return all_errors

# --- Main loop ---
if __name__ == '__main__':
    base_paths = [
        "D:/toronto_microscopy/ixmc/ad_4x_l4440/AD1-cb4037-4x-48hpi_Plate_2643/TimePoint_1/",
        "D:/toronto_microscopy/ixmc/ad_4x_l4440/AD2-cb4037-4x-48hpi_Plate_2642/TimePoint_1/",
        "D:/toronto_microscopy/ixmc/ad_4x_l4440/AD3-cb4037-4x-48hpi_Plate_2641/TimePoint_1/",
    ]
    stitch_paths(base_paths)