# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:07:03 2023

@author: ebjam
"""
import os
import albumentations as A
import cv2
#%%

# Starting at a point where I have already selected the masks I want to augment
interesting_mask_path = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/combo/input/png/train_labels/ori/augment/"
ori_path = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/combo/input/png/train/ori/"

#define augmentations with compose
transformation = A.Compose([
    #Random scale brightnes and contrast
    A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
    A.RandomRotate90(always_apply=False, p=0.5),
    A.VerticalFlip(always_apply=False, p=0.5)
    #Random Flip image
    ])


all_masks = [q for q in os.listdir(interesting_mask_path) if q.endswith(".png")]

for mask in all_masks:
    #Need to load original image
    mask_image = cv2.imread(interesting_mask_path + mask)
    image = cv2.imread(ori_path + mask)
    #Need to augment image
    image_counter = 4
    for w in range(0, image_counter):
        transformed = transformation(image = image, mask = mask_image)
        t_image = transformed['image']
        t_mask = transformed['mask']
        
        save_img_name = ori_path + mask[:-4] + "_augment_" + str(w + 1) + ".png"
        save_mask_name = interesting_mask_path + mask[:-4] + "_augment_" + str(w + 1) + ".png"
        
        cv2.imwrite(save_img_name, t_image)
        cv2.imwrite(save_mask_name, t_mask)
        
#%% resizer
def make_resizer(edgesize):
    resizer = A.Compose([
        A.Resize(always_apply=True, p=1.0, height=edgesize, width=edgesize, interpolation=cv2.INTER_NEAREST)
        ])
    return(resizer)

resizer = make_resizer(800)
#%%
mask_path = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/combo/input/png/train_labels/ori/augment/"
# Make directory for resize
m_r_path = mask_path + "resize/"
if not os.path.exists(m_r_path):
   os.makedirs(m_r_path)

ori_path = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/combo/input/png/train/ori/"
i_r_path = ori_path + "resize/"
if not os.path.exists(i_r_path):
   os.makedirs(i_r_path)

all_masks = [q for q in os.listdir(mask_path) if q.endswith(".png")]

for mask in all_masks:
    mask_image = cv2.imread(interesting_mask_path + mask)
    image = cv2.imread(ori_path + mask)
    
    resized = resizer(image = image, mask = mask_image)
    
    r_image = resized['image']
    r_mask = resized['mask']
    
    save_img_name = i_r_path + mask[:-4] + "resize.png"
    save_mask_name = m_r_path + mask[:-4] + "resize.png"
    
    cv2.imwrite(save_img_name, r_image)
    cv2.imwrite(save_mask_name, r_mask)


#%%
from PIL import Image

m_r_z_path = m_r_path + "zeromapped/"
if not os.path.exists(m_r_z_path):
   os.makedirs(m_r_z_path)
images = [q for q in os.listdir(m_r_path) if q.endswith("png")]
for image in images:
    mask_format = Image.new(mode = 'L', size = (800,800), color=1)
    mask_map = mask_format.load()
    operating_image = Image.open(m_r_path + image)
    load_image = operating_image.load()
    for i in range(0, operating_image.size[0]):
        for j in range(0, operating_image.size[1]):
            pixel_value = load_image[i, j]
            if pixel_value == (0, 0, 0):
                mask_map[i, j] = 0
                print("Boom")
            elif pixel_value == (0, 255, 0):
                mask_map[i,j] = 1
                print("Bam")
            elif pixel_value == (0, 0, 255):
                mask_map[i,j] = 2
                print("Bap")
    mask_format.save(m_r_z_path + image)