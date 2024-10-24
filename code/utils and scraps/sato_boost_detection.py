# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:49:17 2024

@author: ebjam



hAVING A PLAY AROUND WITH sato FILTERS TO SEE HOW USABLE THEY ARE IN MY NEW dapi FIELDS

"""
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import os, cv2
from skimage.filters import sato
import numpy as np
from skimage import io, feature
from skimage.util import img_as_ubyte
from matplotlib.patches import Polygon
#%%

ju1400_1_DAPI = "E:/toronto_microscopy/ixmc/sep_30_success/N2_JU1400_3/dapi/one_field/"

ju_1_dapi_images = [q for q in os.listdir(ju1400_1_DAPI) if q.endswith("TIF")]

for image in ju_1_dapi_images:
    img = cv2.imread(os.path.join(ju1400_1_DAPI, image), -1) #-1 to load as TIF!
    vesselness = sato(img, black_ridges=False)
    
    svesselness_ubyte = img_as_ubyte(vesselness)
    svesselness_ubyte = cv2.blur(svesselness_ubyte, (3,3))
    three_x = svesselness_ubyte * 3
    _, sv_thresh = cv2.threshold(three_x, 5, 255, cv2.THRESH_BINARY)
    color = (255, 255, 255)
    contours, _ = cv2.findContours(sv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = []
    epsilon = 0.001 * cv2.arcLength(contours[0], True)  # Adjust epsilon for smoothing
    for contour in contours:
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Simplify the contour
        simplified_contours.append(approx)
    #cv2.drawContours(img, simplified_contours, -1, color, thickness=cv2.FILLED)
    
    
    
    # Plot original image and processed vesselness side by side
    plt.figure(figsize=(15, 5))

# Original image subplot
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.axis('off')

# Sato filter output subplot
    plt.subplot(1, 2, 2)
    plt.title('Sato Filter Output')
    plt.imshow(svesselness_ubyte)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Image with simplified contours')
    plt.imshow(svesselness_ubyte)
    plt.axis('off')
    # Show the plot
    plt.tight_layout()
    plt.show()
    
#%%
dark_sato = vesselness = sato(img, black_ridges=True)
plt.imshow(dark_sato)
#%%
fivex = svesselness_ubyte * 3
_, fx_thresh = cv2.threshold(fivex, 5, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(fx_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
plt.figure(figsize=(10, 10))
simplified_contours = []
epsilon = 0.001 * cv2.arcLength(contours[0], True)  # Adjust epsilon for smoothing
for contour in contours:
    approx = cv2.approxPolyDP(contour, epsilon, True)  # Simplify the contour
    simplified_contours.append(approx)
    color = [q/255 for q in list(np.random.choice(range(255), size=3))]
    polygon = Polygon(approx[:, 0, :], closed=True, fill=True, fc=color, ec='k', alpha=1)
    plt.gca().add_patch(polygon)
#cv2.drawContours(img, simplified_contours, -1, (115, 115, 0), thickness=cv2.FILLED)
plt.imshow(img)
plt.show()


#%%


print(max(three_x))
#%%
# Define the path to the image directory
ju1400_1_DAPI = "E:/toronto_microscopy/ixmc/sep_30_success/N2_lv4440_cyc1_curve/N2 lv4440 vs cyc1 2mil_Plate_2281/TimePoint_1/dapi/one_field/"
ju1400_1_DY96 = ju1400_1_DAPI.replace("/dapi/", "/dy96/")
ju1400_1_FISH = ju1400_1_DAPI.replace("/dapi/", "/fish/")

chans = [ju1400_1_DAPI, ju1400_1_DY96, ju1400_1_FISH]

dapi_wbw = ju1400_1_DAPI + "worm_by_worm/"
dy96_wbw = ju1400_1_DY96 + "worm_by_worm/"
fish_wbw = ju1400_1_FISH + "worm_by_worm/"

wbws = [dapi_wbw, dy96_wbw, fish_wbw]

for wbw in wbws:
    os.makedirs(wbw, exist_ok=True)
# Add worm by worms here

ju_1_dapi_images = [q for q in os.listdir(ju1400_1_DAPI) if q.endswith("TIF")]
plate_results = []
for image in ju_1_dapi_images:
    img = cv2.imread(os.path.join(ju1400_1_DAPI, image), -1)  # Load the image
    vesselness = sato(img, black_ridges=False)  # Apply Sato filter
    dy96_image = image.replace("w1_combo", "w2_combo")
    dy96_img = cv2.imread(os.path.join(ju1400_1_DY96, dy96_image), -1)
    fish_image = image.replace("w1_combo", "w3_combo")
    fish_img = cv2.imread(os.path.join(ju1400_1_FISH, fish_image), -1)
    # Process the vesselness image
    svesselness_ubyte = img_as_ubyte(vesselness)
    three_x = svesselness_ubyte * 3
    _, sv_thresh = cv2.threshold(three_x, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(sv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_contours = []
    epsilon = 0.001 * cv2.arcLength(contours[0], True)  # Adjust epsilon for smoothing
    namer = 0
    this_well = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            if area < 45000:
                lookup = image + "_worm_" + str(namer)
                measurements = {}
                measurements['image'] = image
                measurements['lookup'] = lookup
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
                mask = np.zeros(img.shape, dtype=np.uint16)
                cv2.drawContours(mask, [contour], -1, 65535, thickness=cv2.FILLED)  # Fill the contour
                # Extract the pixel values within the contour
                masked_image = cv2.bitwise_and(img, mask)
                masked_dimage = cv2.bitwise_and(dy96_img, mask)
                masked_fimage = cv2.bitwise_and(fish_img, mask)
                cropped_dapi = img[y:y+h, x:x+w]
                cropped_dy96 = dy96_img[y:y+h, x:x+w]
                cropped_fish = fish_img[y:y+h, x:x+w]
                cv2.imwrite(dapi_wbw + lookup + ".TIF", cropped_dapi)
                d_lookup = lookup.replace("1_combo.TIF_", "2_combo_")
                cv2.imwrite(dy96_wbw + d_lookup + ".TIF", cropped_dy96)
                f_lookup = lookup.replace("1_combo.TIF_", "3_combo_")
                cv2.imwrite(fish_wbw + f_lookup + ".TIF", cropped_fish)
                crop_mask_dy96 = masked_dimage[y:y+h, x:x+w]
                num_DY96_pixels = len(crop_mask_dy96[crop_mask_dy96 > 5000])
                crop_mask_fish = masked_fimage[y:y+h, x:x+w]
                num_fish_pixels = len(crop_mask_fish[crop_mask_fish > 10000])
                measurements['contour'] = contour
                measurements['dy96_pixels'] = num_DY96_pixels
                measurements['dy96_percent'] = 100 * (num_DY96_pixels / area)
                measurements['fish_pixels'] = num_fish_pixels
                measurements['fish_percent'] = 100 * (num_fish_pixels / area)
                measurements['area_to_perimeter'] = area / perimeter
                #Need to do the worm classification here!! - if it makes sense, then 
                approx = cv2.approxPolyDP(contour, epsilon, True)  # Simplify the contour
                measurements['approximate_contour'] = approx
                simplified_contours.append(approx)
                #this_well.append(measurements)
                plate_results.append(measurements)
                # Could do worm validity prediction here!!
                namer += 1
    #for single_worm in this_well:
     #   plate_results.append(single_worm)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Subplot 1: Original Image
    ax[0].set_title('Original Image')
    ax[0].imshow(img)  # Convert BGR to RGB for proper color display
    ax[0].axis('off')

    # Subplot 2: Three times the vesselness
    ax[1].set_title('Three Times Vesselness')
    ax[1].imshow(three_x, cmap='gray')  # Display the scaled vesselness image
    ax[1].axis('off')

    # Subplot 3: Original Image with Simplified Contours
    ax[2].set_title('Image with Simplified Contours')
    ax[2].imshow(img)  # Display the original image

    # Draw simplified contours as patches
    for approx in simplified_contours:
        area = cv2.contourArea(approx)
        if area > 1000:
            color = [q / 255 for q in np.random.choice(range(256), size=3)]  # Random color for each polygon
            polygon = Polygon(approx[:, 0, :], closed=True, fill=True, fc=color, ec='k', alpha=0.5)
            ax[2].add_patch(polygon)  # Add the polygon patch to the third axes

    ax[2].axis('off')  # Turn off axis numbers and ticks for the third subplot

    # Show the plot
    plt.tight_layout()
    plt.show()



#%%
import pickle

with open(os.path.join(ju1400_1_DAPI, 'ju1400_1&2&3_dapi_sato_unfiltered_oct3.pkl'), 'wb') as file:  # Open a file in binary write mode
    pickle.dump(plate_results, file)  # Serialize the object

#%%
holder = plate_results.copy()

#%%
for dicto in holder:
    dicto['rep'] = 2
#%%
import pandas as pd
rep_2 = pd.DataFrame(holder)
#processed = pd.DataFrame(holder)
#%%
combo = pd.concat([combo, rep_2])

#%%
lv_vs_cyc1 = pd.DataFrame(plate_results)

cyc1 = ['E', 'F', 'G', 'H']

lv_vs_cyc1['line'] = "lv4440"
lv_vs_cyc1['row'] = lv_vs_cyc1['image'].str[24]




