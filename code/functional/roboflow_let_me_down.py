# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:06:38 2024

@author: ebjam
roboflow gives blank images... time to re-associate labels with images!


I want to end up with a similar file structure to yolo/roboflow output, but with images that are not just blank

"""

import os, cv2, shutil

def gen_used_list(path):
    '''
    Parameters
    ----------
    path : String, the path to a folder full of YOLO format labels generated in roboflow

    Returns
    -------
    A list of the images used in generating the dataset, two parts, split on '_png.rf'

    '''
    used_files = [q.split('_png.rf') for q in os.listdir(path)]
    return(used_files)

def ttv_get(base):
    '''
    Parameters
    ----------
    base : base folder for dataset.

    Returns
    -------
    dictionary formatted with handle:path

    '''
    train_path = base + "train/labels/"
    test_path =  base + "test/labels/"
    val_path = base + "valid/labels/"
    sets = {'train':train_path, 'test':test_path, 'valid':val_path}
    return(sets)

def label_mover(sets, dest_sets):
    '''
    

    Parameters
    ----------
    sets : output from ttv_get
    dest_sets : the later defined dest set dicto - ttv_get output but for destinations, not sources

    Returns
    -------
    Transfers labels into correct folders

    '''
    for key, source_path in sets.items():
        dest_path = dest_sets[key] + 'labels/'
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        labels = os.listdir(source_path)
        for label in labels:
            shutil.copy(source_path + label, dest_path + label)

dest_base = "E:/toronto_microscopy/simplifying embryo detection/used_wbw_boost/"
base = "E:/toronto_microscopy/simplifying embryo detection/finalfinalfinal.v2i.yolov8/"
source_base = "E:/toronto_microscopy/simplifying embryo detection/all_wbw_boost/"

def get_dest_sets(dest_root):
    d_test = dest_root + 'test/'
    d_train = dest_root + 'train/' 
    d_val = dest_root + 'valid/'
    dest_sets = {'train':d_train, 'test':d_test, 'valid':d_val}
    for key, dest_directory in dest_sets.items():
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
    return(dest_sets)

dest_sets = get_dest_sets(dest_base)
all_possible_images = [q[:-4] for q in os.listdir(source_base)]
api = [q.replace(".", "-") for q in all_possible_images]
api = [q.replace(" ", "-") for q in api]

for handle, path in ttv_get(base).items():
    used = gen_used_list(path)
    destination = dest_sets[handle] + "images/"
    os.makedirs(destination, exist_ok=True)
    label_dest = dest_sets[handle] + "labels/"
    for image in used:
        if image[0] in api:
            back_to_ori_spaces = image[0].replace('-', ' ')
            back_to_ori_period = back_to_ori_spaces.replace(" TIF", ".TIF")
            source = source_base + back_to_ori_period + ".png"
            img = cv2.imread(source)
            dest = destination + image[0] + "_png.rf" + image[1][-4] + ".jpg"
            cv2.imwrite(dest, img)
            
label_mover(ttv_get(base), get_dest_sets(dest_base))        
