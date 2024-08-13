# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:03:51 2024

@author: ebjam
running predicition on new images:
"""
import joblib, pickle, gzip, os, cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

#%%
loaded_model = joblib.load('C:/Users/ebjam/Documents/GitHub/wormfind/models/worm_detection_classifier/best_random_forest_binary_august01.joblib')
#%%
load_path = r"E:\toronto_microscopy\ixmc\July28LiquidlotN2vsAWR73\July28LiquidlotN2vsAWR73\TimePoint_1\dapi\one_field\binary\N2_AWR73_binary_unfiltered_worms.gz"

with gzip.open(load_path, 'rb') as f:
    binary_particles = pickle.load(f)
    
particle_df = pd.DataFrame(binary_particles)
particle_df['perimeter:area'] = particle_df['Perim.'] / particle_df['Area']
particle_df = particle_df.loc[particle_df['Area'] > 1000].copy()
particle_df = particle_df.loc[particle_df['Area'] < 32000].copy()
important_features = ['Solidity', 'MinFeret', 'Round', 'RawIntDen', 'Perim.', 'Feret', 'IntDen', 'AR', 'Area', 'Circ.', 'perimeter:area']


def predict_on_row(row):
    row_features = pd.DataFrame([row[important_features]])
    return loaded_model.predict(row_features)[0]

particle_df['prediction'] = particle_df.apply(predict_on_row, axis=1)
#%%
images = particle_df['image'].unique()

dapi_image_folder = r"E:/toronto_microscopy/ixmc/July28LiquidlotN2vsAWR73/July28LiquidlotN2vsAWR73/TimePoint_1/dapi/one_field/"
dy_image_folder = r"E:/toronto_microscopy/ixmc/July28LiquidlotN2vsAWR73/July28LiquidlotN2vsAWR73/TimePoint_1/dy96/one_field/"

dapi_list = os.listdir(dapi_image_folder)
dy96_list = os.listdir(dy_image_folder)
for image_with_spaces in dy96_list:
    os.rename(os.path.join(dy_image_folder, image_with_spaces), os.path.join(dy_image_folder, image_with_spaces.replace(" ", "_")))
dy_wbw = dy_image_folder + "worm_by_worm/"
os.makedirs(dy_wbw, exist_ok=True)
for image in images:
    img_segmentations = []
    img_bboxes = []
    tif_image = image.replace('png', 'TIF')
    if tif_image in dapi_list:
        rele_detect = particle_df.loc[particle_df['image'] == image].copy()
        rele_detect = rele_detect.loc[rele_detect['prediction'] == 1].copy()
        img = cv2.imread(os.path.join(dapi_image_folder, tif_image))
        dy_image = tif_image.replace('w1', 'w2')
        dy_img = cv2.imread(os.path.join(dy_image_folder, dy_image), -1)
        for index, row in rele_detect.iterrows():
            raw_segmentation = row['segmentation']
            img_segmentations.append(raw_segmentation)
            x = int(row['BX'])
            y = int(row['BY'])
            width = int(row['Width'])
            height = int(row['Height'])
            name = row['lookup']
            xywhn = [x, y, width, height, name]
            img_bboxes.append(xywhn)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #fig, ax = plt.subplots()
        #fig.set_size_inches(10,10)
        #ax.imshow(img_rgb)
        # Make all segs into polygons
        for seg in img_segmentations:
            random_color = np.random.rand(3)
            random_color = np.append(random_color, 1)
        # Close polygon if necessary
            if seg[0] != seg[-1]:
                seg.append(seg[-1])
            patch = patches.Polygon(seg, closed=True, color=random_color, ec='blue', lw=1)
            #ax.add_patch(patch)
        #plt.title(image)
        #plt.axis('off')
        #plt.show()
        for index, bbox in enumerate(img_bboxes):
            crop = dy_img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            cv2.imwrite(dy_wbw + bbox[4] + ".TIF", crop)

            
            
        
    