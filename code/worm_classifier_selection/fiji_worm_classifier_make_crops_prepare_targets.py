# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 07:59:13 2024

@author: ebjam
"""
import pandas as pd
import os, cv2, gzip, pickle, h5py
import math
import matplotlib.pyplot as plt
import numpy as np

#%%

load_path = r"E:\toronto_microscopy\ixmc\July-9-n2-worm-curve-lot19_Plate_2101\July-9-n2-worm-curve-lot19_Plate_2101\TimePoint_1\dapi\one_field\unfiltered_worms.gz"

with gzip.open(load_path, 'rb') as f:
    loaded_list = pickle.load(f)
    
df = pd.DataFrame(loaded_list)
images = df['image'].unique()
image_path = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dapi/one_field/"
worm_by_worm = image_path + 'worm_by_worm/'
os.makedirs(worm_by_worm, exist_ok=True)
for image in images:
    dapi_img = cv2.imread(os.path.join(image_path, image))
    print(image)
    relevant_detections = df.loc[df['image'] == image]
    for index, row in relevant_detections.iterrows():
        # Fiji gives decimals, now we need to operate at pixels instead of fractions of pixels.
        # To make sure I'm not cropping any of the segmentation out, I'm rounding down x & y, and rounding up H & W to nearest int
        x = math.floor(row['BX']) 
        y = math.floor(row['BY'])
        height = math.ceil(row['Height'])
        width = math.ceil(row['Width'])
        name = row['lookup'].replace(".TIF", "")
        name += ".png"
        this_worm = dapi_img[y:y+height, x:x+width]
        cv2.imwrite(os.path.join(worm_by_worm, name), this_worm)
#%%
# Break here to sort all detections into good and bad folders - will uses these to make target!
#%%
# Making lists from sorted images, and using them to assign target for a classifier.
good = os.listdir("E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dapi/one_field/worm_by_worm/good/")
good_adjust = [q.replace("bo_", "bo.TIF_") for q in good]
good_adjust_2 = [q.replace(".png", "") for q in good_adjust]
found_in_good = [q for q in good_adjust_2 if q in df['lookup'].unique()]

bad = os.listdir("E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dapi/one_field/worm_by_worm/bad/")
bad_adjust = [q.replace("bo_", "bo.TIF_") for q in bad]
bad_adjust_2 = [q.replace(".png", "") for q in bad_adjust]
found_in_bad = [q for q in bad_adjust_2 if q in df['lookup'].unique()]

df['target'] = 1
df.loc[df['lookup'].isin(found_in_bad), 'target'] = 0
#%%
# An additional metric I thought might be intersting would be perimeter/area, I'll make that now

df['perimeter:area'] = df['Perim.'] / df['Area']

# Great! Lets save this df (to p cloud or git hub as well as a harddrive!) and then use it to train some calssifiers

with gzip.open("E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dapi/one_field/unfiltered_labelled_worms.gz", "wb") as f:
    pickle.dump(df, f)
#%%
# Now we need to do the same with the DY96 channel, loading and saving as TIFs

image_path = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/"
worm_by_worm = image_path + 'worm_by_worm/'
os.makedirs(worm_by_worm, exist_ok=True)
for image in images:
    dy_image=image.replace("w1", "w2")
    dy_img = cv2.imread(os.path.join(image_path, dy_image), -1)
    rele_detect = df.loc[df['image'] == image]
    for index, row in rele_detect.iterrows():
        if row['lookup'] in good_adjust_2:
            x = math.floor(row['BX']) 
            y = math.floor(row['BY'])
            height = math.ceil(row['Height'])
            width = math.ceil(row['Width'])
            name = row['lookup'].replace(".TIF", "") + ".tif"
            dy_crop = dy_img[y:y+height, x:x+width]
            cv2.imwrite(os.path.join(worm_by_worm, name), dy_crop)
#%%
# Break to sort some DY channel worms: picked 25 infected, 25 gravid, and 50 both. Took 20% of each class as a testing set.

# Now I need to convert these into h5 files to be a little faster in ilastik:
#%%
path = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/worm_by_worm/seg_training/"
h5_path = path + "h5/"
os.makedirs(h5_path, exist_ok=True)
seg_train_images = [q for q in os.listdir(path) if q.endswith("tif")]
for dy_image in seg_train_images:
    dy_img = cv2.imread(os.path.join(path, dy_image), -1)
    h5_handle = dy_image.replace(".tif", ".h5")
    h5_handle_path = os.path.join(h5_path, h5_handle)
    with h5py.File(h5_handle_path, 'w') as h5file:
        h5file.create_dataset("worm", data=dy_img, dtype=dy_img.dtype)
#%%
path = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/worm_by_worm/seg_training/testing_set/"
h5_path = path + "h5/"
os.makedirs(h5_path, exist_ok=True)
seg_train_images = [q for q in os.listdir(path) if q.endswith("tif")]
for dy_image in seg_train_images:
    dy_img = cv2.imread(os.path.join(path, dy_image), -1)
    h5_handle = dy_image.replace(".tif", ".h5")
    h5_handle_path = os.path.join(h5_path, h5_handle)
    with h5py.File(h5_handle_path, 'w') as h5file:
        h5file.create_dataset("worm", data=dy_img, dtype=dy_img.dtype)

#%%
gt_path = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/worm_by_worm/seg_training/ground_truths/"
ground_truths = [q for q in os.listdir(gt_path) if q.endswith('.png')]
gt_df = pd.DataFrame(ground_truths)

gt_df['image'] = gt_df[0].str[9:-14] + ".TIF"
gt_df['roi'] = gt_df[0].str[-13:-4]
gt_df['lookup'] = gt_df['image'] + '_' + gt_df['roi']
#%%
gt_df_merge = gt_df.merge(df[['lookup', 'tran_seg', 'Area']], on='lookup', how='left')

#%%
pred_path = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/worm_by_worm/seg_training/testing_set/h5/"
gt_images =  list(gt_df_merge[0].unique())
#%%



for image in gt_images:
    pred_handle = image.replace("sporemap_", "")
    pred_handle = pred_handle.replace(".png", "_worm-Simple Segmentation_full_set.png")
    pred_img = cv2.imread(os.path.join(pred_path, pred_handle), -1)
    pred_img[pred_img < 254] = 0
    loaded_gt = cv2.imread(os.path.join(gt_path, image), -1)
    info = gt_df_merge.loc[gt_df_merge[0] == image]
    seg = info['tran_seg'].tolist()[0]
    if isinstance(seg[0], list):
        seg = [tuple(point) for point in seg]
    pts = np.array(seg, np.int32)
    pts = pts.reshape((-1,1,2))
    #print(len(seg))
    mask = np.zeros(loaded_gt.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked_image = cv2.bitwise_and(loaded_gt, loaded_gt, mask=mask)
    specific_color_area = np.sum((masked_image == 255) & (mask == 255))
    cv2.polylines(masked_image, [pts], isClosed=True, color=255)
    plt.imshow(masked_image)
    plt.show()
    percent_area = specific_color_area / info['Area']
    print(float(percent_area))
#%%

#%%
# CHeck file locations and make sure things are in the right place - it looks like you fucked up somewhere.
train_path = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/worm_by_worm/seg_training/"
trainers = [q for q in os.listdir(train_path) if q.endswith("png")]
trainer_truths = [q.replace("sporemap_", "") for q in os.listdir(train_path + "ground_truths/") if q.endswith("png")]
trainer_preds = [q for q in os.listdir(train_path + "h5/") if "Simple Segmentation_full.png" in q]

trainer_pred_handle = [q.replace("-worm_Simple Segmentation_full_set", "") for q in trainer_preds]

mismatch = [q for q in trainer_truths if q not in trainer_pred_handle]
#%%
extra_pred = [q for q in trainer_preds if q not in trainers]
extra_truth = [q for q in trainer_truths if q not in trainers]

#%%
testpath = "E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/worm_by_worm/seg_training/testing_set/"
test_truths = [q.replace("sporemap_", "") for q in os.listdir(testpath + "ground_truth/") if q.endswith('png')]
test_preds = [q for q in os.listdir(testpath + "h5/") if "Simple Segmentation_full.png" in q]

test_pred_handles = [q.replace("_Simple Segmentation_full_set", "") for q in test_preds]

mismatch = [q for q in test_truths if q not in test_pred_handles]

#%%
# From training ground truths!!
gt_sort = gt_images.sort()
preds = [q for q in os.listdir("E:/toronto_microscopy/ixmc/July-9-n2-worm-curve-lot19_Plate_2101/July-9-n2-worm-curve-lot19_Plate_2101/TimePoint_1/dy96/one_field/worm_by_worm/seg_training/h5") if q.endswith("png")]
full_pred = [q for q in preds if "full" in q]
full_pred = [q for q in preds if "Prob" not in q]
full_pred_sort = full_pred.sort()
#%%
for index, image in enumerate(gt_images):
    print(image)
    print(preds[index])
    print("")