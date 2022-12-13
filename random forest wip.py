# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:42:00 2022

@author: ebjam

make this work on a single image first image

then cat training together and test together
then use them as the single in and test for this

worm by worm folder = D:/2022-10-24/correct_dy96/indi_worm/
threshold folder = D:/2022-10-24/correct_dy96/indi_worm/thresholded/
example worm: "D:/2022-10-24/correct_dy96/indi_worm/72hrn2i_25u_worm_1.png"
example gt: "D:/2022-10-24/correct_dy96/indi_worm/thresholded/72hrn2i_25u_worm_1threshold.png"
"""

#%% Imports for random forest

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import cv2
import joblib

#%%Test images

image = cv2.imread("D:/2022-10-24/correct_dy96/indi_worm/72hrn2i_25u_worm_1.png")
gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#%%
training_labels = cv2.imread("D:/2022-10-24/correct_dy96/indi_worm/thresholded/72hrn2i_25u_worm_1threshold.png")
#training labels are 0(background) or 255(microsporidia) - process to 1 = background and 2 for 225
#%%
b,g,r = cv2.split(training_labels) #splits training labels into dimensions, only need one dimension for trianing
#%%wrangling a split channel into having suitable labels. Currently 0, 255, want 1 and 2

b[b == 0] = 1
b[b == 255] = 2
#b is now prepped for the random forest
#%%Need to cat all the features and labels together

#add features to list to store them

#%%z = list of all features

#%%z1 = np.concatenate((z[i] for i in range(0, len(z)), 1)



#%%

sigma_max = 16
sigma_min = 1

#I think I need to explicitly state the features - the new array has way less

features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)

#%%
features = features_func(gimage)
#%%
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=10, max_samples=0.05)
#%%
clf = future.fit_segmenter(b, features, clf)
result = future.predict_segmenter(features, clf)

#%%
#%% Single predictor I guess? but full lab elled image... need to cat and loop all the wat thorugh
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4))
ax[0].imshow(segmentation.mark_boundaries(gimage, result, mode='thick'))
ax[0].contour(b)
ax[0].set_title('Image, mask and segmentation boundaries')
ax[1].imshow(result)
ax[1].set_title('Segmentation')
fig.tight_layout()

#%%
#%%Convert to grey!!!!!!
img_new = cv2.imread("D:/2022-10-24/correct_dy96/indi_worm/72hrn2i_25u_worm_2.png")
gimg_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
features_new = features_func(gimg_new)
#%%Currently works on no-embryos!!
joblib.dump(clf, "C:/Users/ebjam/Documents/GitHub/wormfind/model01_20221213.joblib")
#%%
result_new = future.predict_segmenter(features_new, clf)
#%%
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
ax[0].imshow(segmentation.mark_boundaries(img_new, result_new, mode='thick'))
ax[0].set_title('Image')
ax[1].imshow(result_new)
ax[1].set_title('Segmentation')
fig.tight_layout()

plt.show()