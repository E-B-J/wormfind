# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:57:05 2024

@author: ebjam

whole run, worms to fish
"""

import os
import cv2
import numpy as np
from skimage import io, feature
from skimage.filters import frangi, sato
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from scipy import ndimage
import joblib
from functools import partial

input_folder = 'E:/toronto_microscopy/ixmc/Sep16_detergent_tests/Sep16_p2_detergent_test/TimePoint_1/dapi/one_field/'
dy96_folder = 'E:/toronto_microscopy/ixmc/Sep16_detergent_tests/Sep16_p2_detergent_test/TimePoint_1/dy96/one_field/'
fish_folder = 'E:/toronto_microscopy/ixmc/Sep16_detergent_tests/Sep16_p2_detergent_test/TimePoint_1/fish/one_field/'
output_folder = input_folder + "vesselness/"
worm_by_worm = input_folder + "worm_by_worm/"
d_worm_by_worm = dy96_folder + "worm_by_worm/"
f_worm_by_worm = fish_folder + "worm_by_worm/"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(worm_by_worm, exist_ok=True)
os.makedirs(d_worm_by_worm, exist_ok=True)
os.makedirs(f_worm_by_worm, exist_ok=True)
detections = []
figure_size = (10,10)
figure_dpi = 100

worm_clf_path = r"C:\Users\ebjam\Documents\GitHub\wormfind\models\worm_detection_classifier\sato_worm_detection_predictor.joblib"
worm_clf = joblib.load(worm_clf_path)
fish_clf_path = "E:/toronto_microscopy/ixmc/Aug 3 Slow Curve_Plate_2142/Aug 3 Slow Curve_Plate_2142/TimePoint_1/fish/best_rf_aug27.joblib"
fish_clf = joblib.load(fish_clf_path)
sig_min = 1
sig_max = 16
features_func = partial(
    feature.multiscale_basic_features,
    intensity = True,
    edges = True,
    texture = True,
    sigma_min = sig_min,
    sigma_max = sig_max,
    )

kernel = np.ones((3, 3), np.uint8)

def unrotate(fish_prediction_img, clahe_cmf, rotate_45_dims):
    ori_x, ori_y = clahe_cmf.shape[0:2]
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

clahe = cv2.createCLAHE(clipLimit=5)
num_to_do = len([q for q in os.listdir(input_folder) if q.endswith("TIF")])
tracker = 1
for filename in os.listdir(input_folder):
    if filename.endswith("TIF"):
        
        this_well = []
        print(filename)
        file_path = os.path.join(input_folder, filename)
        dy_filename = filename.replace("w1_combo", "w2_combo")
        fish_filename = filename.replace("w1_combo", "w3_combo")
        image = cv2.imread(file_path, -1)
        dimage = cv2.imread(dy96_folder + dy_filename, -1)
        fimage = cv2.imread(fish_folder + fish_filename, -1)
        # Apply the Sato Tubeness - ## We're pretending each worm is a tiny bloodvessel, don't let the code know they're actually worms
        # Ahem, Take a look at these blood vessels computer! Aren't they cool?
        svesselness = sato(image, black_ridges=False)
        svesselness_ubyte = img_as_ubyte(svesselness)
        svesselness_ubyte = cv2.blur(svesselness_ubyte, (3,3))
        _, sv_thresh = cv2.threshold(svesselness_ubyte, 10, 255, cv2.THRESH_BINARY)
        color = (255, 255, 255)
        contours, _ = cv2.findContours(sv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, thickness=cv2.FILLED)
        #fig, ax = plt.subplots(figsize=figure_size, dpi=figure_dpi)
        #ax.imshow(image)
        namer = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                measurements = {}
                measurements['image'] = filename
                lookup = filename + "_worm_" + str(namer)
                measurements['lookup'] = lookup
                color = [q/255 for q in list(np.random.choice(range(255), size=3))]
                polygon = Polygon(contour[:, 0, :], closed=True, fill=True, fc=color, ec='k', alpha=1)
                #ax.add_patch(polygon)
                measurements['area'] = area
                perimeter = cv2.arcLength(contour, True)
                measurements['perimeter'] = perimeter
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                measurements['circularity'] = circularity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                measurements['solidity'] = solidity
                bbox = cv2.boundingRect(contour)
                measurements['bbox'] = bbox
                x, y, w, h = bbox
                mask = np.zeros(image.shape, dtype=np.uint16)
                cv2.drawContours(mask, [contour], -1, 65535, thickness=cv2.FILLED)  # Fill the contour
                # Extract the pixel values within the contour
                masked_image = cv2.bitwise_and(image, mask)
                masked_dimage = cv2.bitwise_and(dimage, mask)
                masked_fimage = cv2.bitwise_and(fimage, mask)
                cropped_dapi = masked_image[y:y+h, x:x+w]
                cropped_masked_dimage = masked_dimage[y:y+h, x:x+w]
                cropped_masked_fimage = masked_fimage[y:y+h, x:x+w]
                d_lookup = lookup.replace("1_combo.TIF_", "2_combo_")
                cv2.imwrite(d_worm_by_worm + d_lookup + ".TIF", cropped_masked_dimage)
                f_lookup = lookup.replace("1_combo.TIF_", "3_combo_")
                cv2.imwrite(f_worm_by_worm + d_lookup + ".TIF", cropped_masked_fimage)
                num_fish_pixels = len(cropped_masked_fimage[cropped_masked_fimage > 13000])
                #plt.imshow(cropped_masked_fimage)
                #plt.show()
                pixel_values = cropped_dapi[cropped_dapi > 0]
                # Get only non-zero pixel values
                mean_value = np.mean(pixel_values)
                #print(mean_value)
                max_val = max(pixel_values)
                min_val = min(pixel_values)
                measurements['mean'] = mean_value
                measurements['max'] = max_val
                measurements['min'] = min_val
                measurements['dapi_std'] = np.std(pixel_values)
                measurements['contour'] = contour
                measurements['fish_pixels'] = num_fish_pixels
                measurements['fish_percent'] = 100 * (num_fish_pixels / area)
                measurements['area_to_perimeter'] = area / perimeter
                #Need to do the worm classification here!! - if it makes sense, then 
                this_well.append(measurements)
                # Could do worm validity prediction here!!
                namer += 1
        # Turn this into a function

        well_df = pd.DataFrame(this_well)
        # Semi-error handling for no detections
        if len(well_df) > 0:
            variables_for_random_forest = ['area', 'perimeter', 'circularity', 'solidity', 'mean', 'dapi_std', 'area_to_perimeter']
            for feature in variables_for_random_forest:
                feature_list = well_df[feature].tolist()
                mean = np.mean(feature_list)
                std = np.std(feature_list)
                well_df[feature + '_well_zscore'] = (well_df[feature] - mean) / std
                well_df[feature + '_well_zscore'] = well_df[feature + '_well_zscore'].fillna(0)
            z_scores = [q + '_well_zscore' for q in variables_for_random_forest] 
            full_forest = variables_for_random_forest + z_scores
            well_to_predict = well_df[full_forest]
            worm_status = worm_clf.predict(well_to_predict)
            well_df['worm_prediction'] = worm_status 
            well_df['aspect_ratio'] = 0
            well_df['rotated'] = False
            well_df['#sporecategory'] = 0
            well_df['dirty_fish%'] = 0
            well_df['rotate_45_dim'] = 0
            # Loop fucks here!
            to_fish_for = well_df.loc[well_df['worm_prediction'] == 1]
            for index, row in to_fish_for.iterrows():
                look_up_string = row['lookup']
                updater = well_df.loc[well_df['lookup'] == look_up_string]
    
                x, y, w, h = row['bbox']
                cont = row['contour']
                # Have to reset the mask each time to not crop out each successive worm. Although......
                mask = np.zeros(fimage.shape, dtype=np.uint16)
                cv2.drawContours(mask, [cont], -1, 65535, thickness=cv2.FILLED)  # Fill the contour
                masked_fimage = cv2.bitwise_and(fimage, mask)
                fworm = masked_fimage[y:y+h, x:x+w]
                # Everything above this line is unfucked
                # Now I need to clahe and size adjust....
                clahe_fworm = clahe.apply(fworm) + 30
                make_ar = [w, h]
                aspect_ratio = max(make_ar) / min(make_ar)
                well_df.loc[well_df['lookup'] == look_up_string, 'aspect_ratio'] = aspect_ratio
                if aspect_ratio > 3:
                    well_df.loc[well_df['lookup'] == look_up_string, 'rotated'] = True
                    rotate_45 = ndimage.rotate(clahe_fworm, 45, reshape = True, order = 0)
                    rotate_45_dim = rotate_45.shape[0]
                    well_df.loc[well_df['lookup'] == look_up_string, 'rotate_45_dim'] = rotate_45_dim
                    resized_cfworm = cv2.resize(rotate_45, (320, 320))
                else:
                    resized_cfworm = cv2.resize(clahe_fworm, (320,320))
                # Great, now extract waaaayyyyyyyy too many features:
                features = features_func(resized_cfworm)
                X = features.reshape(-1, features.shape[-1])
                fish_prediction = fish_clf.predict(X)
                fish_prediction_img = fish_prediction.reshape((320,320))
                # Resize
                if well_df.loc[well_df['lookup'] == look_up_string, 'rotated'].iloc[0] == True:
                    r45_dims = [well_df.loc[well_df['lookup'] == look_up_string, 'rotate_45_dim'].iloc[0], well_df.loc[well_df['lookup'] == look_up_string, 'rotate_45_dim'].iloc[0]]
                    fish_prediction_img_o_size = unrotate(fish_prediction_img, clahe_fworm, r45_dims)
                elif well_df.loc[well_df['lookup'] == look_up_string, 'rotated'].iloc[0] == False:
                    fish_prediction_img_o_size = cv2.resize(fish_prediction_img, (w, h))
                
                num_pixels_253 = np.sum(fish_prediction_img_o_size == 253)
                well_df.loc[well_df['lookup'] == look_up_string, '#sporecategory'] = num_pixels_253
                well_df.loc[well_df['lookup'] == look_up_string, 'dirty_fish%'] = num_pixels_253 / row['area']
                mask = fish_prediction_img_o_size == 253
                binary_mask = mask.astype(np.uint8)
                eroded_binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
                #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                #ax[0].imshow(fworm, cmap='gray')
                #ax[0].set_title("Contrast adjusted input image")
                #ax[0].axis("off")
                
                #for q in cont:
                #    trans_x = [q[0][0] - x for q in cont]
                #    trans_y = [q[0][1] - y for q in cont]
                #contour_points = np.vstack((trans_x, trans_y)).T
                #contour_points = contour_points.reshape((-1, 1, 2))
                #mask2 = eroded_binary_mask * 255
                
                #cv2.drawContours(mask2, [contour_points], -1, 126, thickness = 2)
                #ax[1].imshow(mask2)
                #ax[1].set_title("FISH infection prediction")
                #ax[1].axis("off")
                #plt.tight_layout()
                #plt.show()
                    
            
            
            
            loop_pass = well_df.to_dict(orient='records')
            for entry in loop_pass:
                detections.append(entry)
            print("Completed image " + str(tracker) + " / " + str(num_to_do))
            tracker += 1
            #plt.show()
            
        
#%%

x, y, w, h = cv2.boundingRect(cont)

for q in cont:
    trans_x = [q[0][0] - x for q in cont]
    trans_y = [q[0][1] - y for q in cont]
contour_points = np.vstack((trans_x, trans_y)).T
contour_points = contour_points.reshape((-1, 1, 2)) 

#%%
plt.imshow(clahe_fworm)
cont_transposed = []
cv2.drawContours(clahe_fworm, [contour_points], 0, 65355, thickness = 3)
#plt.imshow(fworm)
plt.show()
        #%%
                # Here I need to do CLAHE and resize!
                clahe = cv2.createCLAHE(clipLimit=5)
                clahe_cmf = clahe.apply(cropped_masked_fimage) + 30
                make_ar = [w, h]
                aspect_ratio = max(make_ar) / min(make_ar)
                measurements['aspect_ratio'] = aspect_ratio
                if aspect_ratio > 3:
                    measurements['rotated'] = True
                    rotate_45 = ndimage.rotate(clahe_cmf, 45, reshape = True, order = 0)
                    rotate_45_dims = [rotate_45.shape[0], rotate_45.shape[1]]
                    resized_cmf = cv2.resize(rotate_45, (320, 320))
                    
                else:
                    measurements['rotated'] = False
                    resized_cmf = cv2.resize(clahe_cmf, (320,320))
                # Feature_extraction and prediction!
                # Make this a function
                features = features_func(resized_cmf)
                X = features.reshape(-1, features.shape[-1])
                fish_prediction = fish_clf.predict(X)
                # Can cheat a little here:
                # Get the number of pixels classified as FISH signal,
                # and use that as a funciton of area to get infection level 
                # - just in case the image mending doesn't work!
                num_pixels_253 = np.sum(fish_prediction == 253)
                measurements['internal_predicted_fish_pixels'] = num_pixels_253
                measurements['dirty_fish%'] = num_pixels_253 / area
                
                fish_prediction_img = fish_prediction.reshape((320,320))
                # Unrotation if necessary
                if measurements['rotated'] == True:
                    orig_size_fish_prediction = unrotate(fish_prediction_img, clahe_cmf)
                    # Work out rest of unrotation here!
                elif measurements['rotated'] == False:
                    orig_size_fish_prediction = fish_prediction_img.reshape(clahe_cmf.shape[1], (clahe_cmf.shape[0]))
                # Crop both to mask of worm?
                
                cropped_image = image[y:y+h, x:x+w]
                cropped_dy96 = dimage[y:y+h, x:x+w]
                cv2.imwrite(worm_by_worm + lookup + ".png", cropped_image)
                cv2.imwrite(d_worm_by_worm + lookup + "dy96.png", cropped_dy96)
                cv2.imwrite(f_worm_by_worm + lookup + "fish.png", cropped_masked_fimage)
                cv2.imwrite(f_worm_by_worm + lookup + "fish_prediction.png", orig_size_fish_prediction)


#%%
ultra_df = pd.DataFrame(detections)
#%%
ultra_df['col'] = ultra_df['image'].str[18:20]
ultra_df['col'] = ultra_df['col'].str.replace("0", "").astype(int)
ultra_df['row'] = ultra_df['image'].str[17]

N2 = ['A', 'B', 'C', 'D']

ultra_df['line'] = 'AWR73'
ultra_df.loc[ultra_df['row'].isin(N2), 'line'] = "N2"

#%%
import seaborn as sns
ultra_worms = ultra_df.loc[ultra_df['worm_prediction'] == 1].copy()
sns.violinplot(x='col', y='dirty_fish%', cut=0, inner=None, data=ultra_worms, hue='line', split=True)
sns.pointplot(x='col', y='dirty_fish%', join=False, hue='line', dodge=0.3, palette='dark', data=ultra_worms)

#%%
with open(os.path.join(input_folder,'n2_lv4440_2mNp_sep16_p2.pkl'), 'wb') as file:
    pickle.dump(detections, file)

#%%
print(len(well_df))


#%%
n2_vs_awr73 = []
#%%
for q in detections:
    n2_vs_awr73.append(q)
#%%
det_plate_1 = det_1_upto_g.copy()
#%%
for detection in detections:
    detection["plate_#"] = 2
#%%
for dicto in det_plate_1:
    dicto['plate_#'] = 1
    
    
#%%

reps_df = pd.DataFrame(det_plate_1)
#%%
def spore_converter_dict():
    dicto = {}
    for i in range(11):
        col = i + 1
        spores = 2000000 / 2**i
        dicto[col] = spores
    dicto[12] = 0
    return(dicto)

test_1 = spore_converter_dict()

def spore_converter(col):
    spore = test_1[col]
    return(spore)


#%%

reps_df['col'] = reps_df['image'].str[23:25]

def lead_strip(string):
    if string.startswith("0"):
        num_string = int(string.replace("0", ""))
    else:
        num_string = int(string)
    return(num_string)

reps_df['num_col'] = reps_df['col'].apply(lead_strip)
reps_df['row'] = reps_df['image'].str[22]
n2 = ['A', 'B', 'C', 'D']
parisii = ['A', 'B', 'E', 'F']
reps_df['worm_line'] = 'JU1400'
reps_df['spore_line'] = 'ironsii'

reps_df.loc[reps_df['row'].isin(n2), 'worm_line'] = "N2"
reps_df.loc[reps_df['row'].isin(parisii), 'spore_line'] = "parisii"
reps_df['spores'] = reps_df['num_col'].apply(spore_converter)

reps_df['line_line'] = reps_df['worm_line'] + "_" + reps_df['spore_line']
reps_df['spore_line_line'] = reps_df['line_line'] + "_" + reps_df['spores'].astype(str)


#%%
columns_of_interest = ['line_line', 'spores', 'plate_#', "worm_line", "spore_line", "dirty_fish%", "area"]

# Group by 'replicate' and calculate mean and standard deviation
grouped_mean = df.groupby('replicate')[columns_of_interest].mean().reset_index()
grouped_std = df.groupby('replicate')[columns_of_interest].std().reset_index()

# Optional: Combine mean and std into a single DataFrame
grouped_stats = pd.merge(grouped_mean, grouped_std, on='replicate', suffixes=('_mean', '_std'))




#%%

parisii_only = reps_df.loc[reps_df['spore_line'] == 'parisii']
parisii_only = parisii_only.loc[parisii_only['worm_prediction'] == 1].copy()
np_n2 = parisii_only.loc[parisii_only['worm_line'] == 'N2'].copy()
np_n2['col_off'] = np_n2['num_col']
np_ju = parisii_only.loc[parisii_only['worm_line'] == 'JU1400'].copy()
np_ju['col_off'] = np_ju['num_col'] + 0.15
sns.violinplot(x='col', y='dirty_fish%', cut=0, inner=None, data=parisii_only, hue='worm_line', split=True)
sns.pointplot(x='col_off', y='dirty_fish%', join=False, style='plate_#', hue='plate_#', palette='dark', data=np_n2)
sns.pointplot(x='col_off', y='dirty_fish%', join=False, style='plate_#',  hue='plate_#', palette='dark', data=np_ju)

#%%
plates = [1, 2, 3]
marks = ['D', 'O', '^']
for plate in plates:
    
    rep = reps_df.loc[reps_df['plate_#'] == plate].copy()
    parisii = rep.loc[rep['spore_line'] == "parisii"].copy()
    ironsii = rep.loc[rep['spore_line'] == "ironsii"].copy()
    grouped = rep.groupby(['spore_line', 'worm_line', 'spores'])
    sns.pointplot(x='num_col', y="dirty_fish%", data=parisii, hue = "worm_line", dodge=0.3, split=True, join=False, markers=marks[plate-1], alpha=0.5)
    

#%%
columns_of_interest = ["dirty_fish%", "area"]
grouped = reps_df.groupby(['plate_#', 'spore_line', 'worm_line', 'spores'])[columns_of_interest].mean().reset_index()
g_grouped = grouped.groupby(['spore_line', 'worm_line', 'spores'])[["dirty_fish%", "area"]].mean().reset_index()
ironsii = grouped.loc[grouped['spore_line'] == 'ironsii']
parisii = grouped.loc[grouped['spore_line'] == 'parisii']
g_ironsii = g_grouped.loc[g_grouped['spore_line'] == 'ironsii']
g_parisii = g_grouped.loc[g_grouped['spore_line'] == 'parisii']

spore_lines = []
#%%
fig, ax = plt.subplots(2, 1, figsize=(16, 8))
marks = ['D', 'o', '^']
for plate in plates:
    i_rep = ironsii.loc[ironsii['plate_#'] == plate]
    p_rep = parisii.loc[parisii['plate_#'] == plate]
    mega_rep = reps_df.loc[reps_df['plate_#'] == plate]
    mega_rep_i = mega_rep.loc[mega_rep['spore_line'] == 'ironsii']
    mega_rep_p = mega_rep.loc[mega_rep['spore_line'] == 'parisii']
    sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=i_rep, palette={"N2": "#f2b124", "JU1400": "#2458f2"}, join=False, markers=marks[plate-1], alpha = 0.5, ax=ax[0], zorder=10)
    sns.stripplot(x='spores', y='dirty_fish%', hue='worm_line',dodge = 'worm_line', data=mega_rep_i, palette={"N2": "#f2b124", "JU1400": "#2458f2"}, alpha = 0.2, marker = marks[plate-1], ax=ax[0], zorder = 9)
    sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=p_rep, palette={"N2": "#f2b124", "JU1400": "#2458f2"}, join=False, markers=marks[plate-1], alpha = 0.5, ax=ax[1], zorder = 1)
    sns.stripplot(x='spores', y='dirty_fish%', hue='worm_line',dodge = 'worm_line', data=mega_rep_p, palette={"N2": "#f2b124", "JU1400": "#2458f2"}, alpha = 0.2, marker = marks[plate-1], ax=ax[1], zorder = 0)
g_ironsii_n2 = g_ironsii.loc[g_ironsii['worm_line'] == 'N2']
g_ironsii_ju = g_ironsii.loc[g_ironsii['worm_line'] == 'JU1400']
g_parisii_n2 = g_parisii.loc[g_parisii['worm_line'] == 'N2']
g_parisii_ju = g_parisii.loc[g_parisii['worm_line'] == 'JU1400']

sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_ironsii_n2, palette={"N2": "#c7580e"}, join=False, ax=ax[0], zorder = 5)
sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_ironsii_ju, palette={"JU1400": "#042485"}, join=False, ax=ax[0], zorder = 5)

sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_parisii_n2, palette={"N2": "#c7580e"}, join=False, ax=ax[1], zorder = 9)
sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_parisii_ju, palette={"JU1400": "#042485"}, join=False, ax=ax[1], zorder = 9)
sns.despine()
#sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_parisii, palette='dark', errorbar=None)
#%%[ine', 'spores', 'plate_#', "worm_line", "spore_line", "dirty_fish%", "area"]

plates = [1, 2, 3]
for plate in plates:
    rep = reps_df.loc[reps_df['plate_#'] == plate].copy()

    
    rep_grouped_mean = rep_1.groupby('spore_line_line').mean().reset_index()
    rep_group_mean['rep'] = rep


rep_grouped_mean = rep_1.groupby('spore_line_line').mean().reset_index()


#%%

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(16, 8))
marks = ['D', 'o', '^']

for plate in plates:
    i_rep = ironsii.loc[ironsii['plate_#'] == plate]
    p_rep = parisii.loc[parisii['plate_#'] == plate]
    mega_rep = reps_df.loc[reps_df['plate_#'] == plate]
    mega_rep_i = mega_rep.loc[mega_rep['spore_line'] == 'ironsii']
    mega_rep_p = mega_rep.loc[mega_rep['spore_line'] == 'parisii']
    
    sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=i_rep, 
                  palette={"N2": "#f2b124", "JU1400": "#2458f2"}, join=False, 
                  markers=marks[plate-1], alpha=0.5, ax=ax[0],edgecolor='k', edgewidth=3, zorder=1)
    
    sns.stripplot(x='spores', y='dirty_fish%', hue='worm_line', dodge='worm_line', 
                  data=mega_rep_i, palette={"N2": "#f2b124", "JU1400": "#2458f2"}, 
                  alpha=0.2, marker=marks[plate-1], ax=ax[0], zorder=5)
    
    sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=p_rep, 
                  palette={"N2": "#f2b124", "JU1400": "#2458f2"}, join=False, 
                  markers=marks[plate-1], alpha=0.5, ax=ax[1],edgecolor='k',edgewidth=3, zorder=1)
    
    sns.stripplot(x='spores', y='dirty_fish%', hue='worm_line', dodge='worm_line', 
                  data=mega_rep_p, palette={"N2": "#f2b124", "JU1400": "#2458f2"}, 
                  alpha=0.2, marker=marks[plate-1], ax=ax[1], zorder=5)

g_ironsii_n2 = g_ironsii.loc[g_ironsii['worm_line'] == 'N2']
g_ironsii_ju = g_ironsii.loc[g_ironsii['worm_line'] == 'JU1400']
g_parisii_n2 = g_parisii.loc[g_parisii['worm_line'] == 'N2']
g_parisii_ju = g_parisii.loc[g_parisii['worm_line'] == 'JU1400']

sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_ironsii_n2, 
              palette={"N2": "#c7580e"}, join=False, ax=ax[0], zorder=100, scale = 2)

sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_ironsii_ju, 
              palette={"JU1400": "#042485"}, join=False, ax=ax[0], zorder=101, scale = 2)

sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_parisii_n2, 
              palette={"N2": "#c7580e"}, join=False, ax=ax[1], zorder=100, scale=2)

sns.pointplot(x='spores', y='dirty_fish%', hue='worm_line', data=g_parisii_ju, 
              palette={"JU1400": "#042485"}, join=False, ax=ax[1], zorder=101, scale = 2)

sns.despine()
ax[0].get_legend().remove()
ax[1].get_legend().remove()
plt.show()

#%%
ju1400 = reps_df.loc[reps_df['worm_line'] == "JU1400"]
#%%
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
for plate in plates:
    
    i_rep = ironsii.loc[ironsii['plate_#'] == plate]
    p_rep = parisii.loc[parisii['plate_#'] == plate]
    mega_rep = reps_df.loc[reps_df['plate_#'] == plate]
    mega_rep = mega_rep.loc[mega_rep['worm_line'] == 'JU1400']
    mega_rep_i = mega_rep.loc[mega_rep['spore_line'] == 'ironsii']
    mega_rep_p = mega_rep.loc[mega_rep['spore_line'] == 'parisii']
    i_rep = i_rep.loc[i_rep['worm_line'] == 'JU1400']
    p_rep = p_rep.loc[p_rep['worm_line'] == 'JU1400']
    sns.pointplot(x='spores', y='dirty_fish%', join = False, data = i_rep, color='pink', ax=ax)
    sns.pointplot(x='spores', y='dirty_fish%', join = False, data = p_rep, color = 'purple', ax=ax)
sns.stripplot(x='spores', y='dirty_fish%', data = ju1400, hue = 'spore_line', dodge = 'spore_line', palette = {"ironsii": 'pink', "parisii": 'purple'}, alpha = 0.2)
sns.pointplot(x='spores', y='dirty_fish%', data = g_ironsii, color = 'deeppink')
sns.pointplot(x='spores', y='dirty_fish%', data = g_parisii, color = 'blueviolet')
sns.despine()
#%%

g_ironsii_n2 = g_ironsii.loc[g_ironsii['worm_line'] == 'N2']
g_ironsii_ju = g_ironsii.loc[g_ironsii['worm_line'] == 'JU1400']
g_parisii_n2 = g_parisii.loc[g_parisii['worm_line'] == 'N2']
g_parisii_ju = g_parisii.loc[g_parisii['worm_line'] == 'JU1400']

sns.pointplot(x='spores', y='dirty_fish%', hue='spore_line', data=g_ironsii_n2, 
              palette={"N2": "#c7580e"}, join=False, ax=ax, zorder=100, scale = 2)

sns.pointplot(x='spores', y='dirty_fish%', hue='spore_line', data=g_ironsii_ju, 
              palette={"JU1400": "#042485"}, join=False, ax=ax, zorder=101, scale = 2)

sns.despine()
ax[0].get_legend().remove()
ax[1].get_legend().remove()
plt.show()

#%%

with open(os.path.join(input_folder,'n2_vs_awr_reps.pkl'), 'wb') as f:  # open a text file
    pickle.dump(n2_vs_awr73, f) # serialize the list
    
#%%
n2_vs_awr_df = pd.DataFrame(n2_vs_awr73)
nva_worms = n2_vs_awr_df.loc[n2_vs_awr_df['worm_prediction'] == 1].copy()
nva_worms['col'] = nva_worms['image'].str[22:24]
nva_worms['row'] = nva_worms['image'].str[21]

nva_worms['line'] = 'AWR73'
n2 = ['A', 'B', 'C', 'D']
nva_worms.loc[nva_worms['row'].isin(N2), 'line'] = "N2"
nva_worms['num_col'] = nva_worms['col'].apply(lead_strip)
nva_worms['spores'] = nva_worms['num_col'].apply(spore_converter)
#%%
plates = [1, 2, 3]
marks = ['D', 'o', '^']
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for plate in plates:
    rep = nva_worms.loc[nva_worms['plate_#'] == plate]
    sns.stripplot(x='spores', y='dirty_fish%', data=rep, hue='line', dodge = 'line', marker = marks[plate-1], alpha = 0.5, palette =  {"N2": "#f29252", "AWR73": "#3b63db"}, zorder = 0, ax=ax)

for plate in plates:
    rep = nva_worms.loc[nva_worms['plate_#'] == plate]
    sns.pointplot(x='spores', y='dirty_fish%', data=rep,dodge = 0.4, hue='line', join=False, palette = {"N2": "#db7a39", "AWR73": "#2953cf"}, markers = marks[plate-1], alpha=0.2, zorder = 1, ax=ax)


sns.pointplot(x='spores', y='dirty_fish%', data=nva_worms, hue='line', dodge = 0.1,  palette = {"N2": "#c7580e", "AWR73": "#042485"}, join=False, scale=2, zorder = 100, ax=ax, markers = 'x')

ax.get_legend().remove()
sns.despine()

#%%
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
for plate in plates:
    rep = nva_worms.loc[nva_worms['plate_#'] == plate]
    rep_n2 = rep.loc[rep['line'] == 'N2']
    rep_n2_worm_count = rep_n2.groupby(['line', 'spores']).size().reset_index(name='counts')
    sns.pointplot(x='spores', y='counts', data=rep_n2_worm_count, hue='line', join=False, palette = {"N2": "#db7a39", "AWR73": "#2953cf"}, markers = marks[plate-1], alpha=0.2, zorder = 1, ax=ax)
    rep_AW = rep.loc[rep['line'] == 'AWR73']
    rep_AW_worm_count = rep_AW.groupby(['line', 'spores']).size().reset_index(name='counts')
    sns.pointplot(x='spores', y='counts', data=rep_AW_worm_count, hue='line', join=False, palette = {"N2": "#db7a39", "AWR73": "#2953cf"}, markers = marks[plate-1], alpha=0.2, zorder = 1, ax=ax)


n2 = nva_worms.loc[nva_worms['line'] == "N2"]
n2_count = n2.groupby(['spores', 'plate_#']).size().reset_index(name='counts')
aw = nva_worms.loc[nva_worms['line'] == "AWR73"]
aw_count = aw.groupby(['spores', 'plate_#']).size().reset_index(name='counts')

sns.pointplot(x='spores', y='counts', data = n2_count, color='orange', join=False, scale = 2)
sns.pointplot(x='spores', y='counts', data = aw_count, color='blue', join=False, scale = 2)
sns.despine()

#%%
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
for plate in plates:
    rep = reps_df.loc[reps_df['plate_#'] == plate]
    rep_n2 = rep.loc[rep['worm_line'] == 'N2']
    rep_n2_worm_count = rep_n2.groupby(['worm_line', 'spore_line', 'spores']).size().reset_index(name='counts')
    sns.pointplot(x='spores', y='counts', data=rep_n2_worm_count, hue='spore_line', join=False, palette = {"parisii": "#db7a39", "ironsii": "#2953cf"}, markers = marks[plate-1], alpha=0.2, zorder = 1, ax=ax)
n2_ = reps_df.loc[reps_df['worm_line'] == 'N2']
n2_count_spore = n2_.groupby(['worm_line', 'spore_line', 'spores', 'plate_#']).size().reset_index(name='counts')
sns.pointplot(x='spores', y='counts', hue = 'spore_line', data = n2_count_spore, join=False, scale = 2)

sns.despine()
#%% 
    
 rep_AW = rep.loc[rep['line'] == 'AWR73']
    rep_AW_worm_count = rep_AW.groupby(['line', 'spores']).size().reset_index(name='counts')
    sns.pointplot(x='spores', y='counts', data=rep_AW_worm_count, hue='line', join=False, palette = {"N2": "#db7a39", "AWR73": "#2953cf"}, markers = marks[plate-1], alpha=0.2, zorder = 1, ax=ax)


n2 = nva_worms.loc[nva_worms['line'] == "N2"]
n2_count = n2.groupby(['spores', 'plate_#']).size().reset_index(name='counts')
aw = nva_worms.loc[nva_worms['line'] == "AWR73"]
aw_count = aw.groupby(['spores', 'plate_#']).size().reset_index(name='counts')

sns.pointplot(x='spores', y='counts', data = n2_count, color='orange', join=False, scale = 2)
sns.pointplot(x='spores', y='counts', data = aw_count, color='blue', join=False, scale = 2)
sns.despine()



#%%
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ju1400 = reps_df.loc[reps_df['worm_line'] == "JU1400"]
for plate in plates:
    rep = ju1400.loc[ju1400['plate_#'] == plate]
    sns.pointplot(x='spores', y='dirty_fish%', data = rep, hue='spore_line', dodge = 0.3, markers = marks[plate-1], palette={"parisii":'#e3adea',"ironsii": "#96ddd8"}, alpha = 0.2, join=False, ax=ax)
#sns.pointplot(x='spores', y='dirty_fish%', data = ju1400, hue='spore_line', markers = 'D', color = 'k', join=False, scale = 2.05, ax=ax, edgecolor = 'k', linewidth = 0)
sns.pointplot(x='spores', y='dirty_fish%', data = ju1400, hue='spore_line', markers = '*', palette = {'parisii': "#e52afe", 'ironsii': '#00ffed'}, join=False, scale = 2, ax=ax, edgecolor = 'k', linewidth = 1)
#sns.pointplot(x='spores', y='dirty_fish%', data = ju1400, hue='spore_line', markers = 'o', palette = {'parisii': "#e52afe", 'ironsii': '#00ffed'}, join=False, scale = 0.5, ax=ax, edgecolor = 'k', linewidth = 1)
sns.despine()
ax.get_legend().remove()