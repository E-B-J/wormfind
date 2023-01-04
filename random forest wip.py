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
import os #accessing files
import numpy as np #image and number handling
import matplotlib.pyplot as plt #showing images
from skimage import data, segmentation, feature, future #RF tools
from sklearn.ensemble import RandomForestClassifier # RF
from functools import partial #Feature gen
from math import ceil
import cv2 #Image handling
import joblib #Saving/loading model

#%%Test images

image2 = cv2.imread("D:/2022-10-24/correct_dy96/indi_worm/72hrn2i_25u_worm_2.png")
gimage2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
x, y = gimage.shape
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
#%%
def images_and_truths(path):
    '''
    

    Parameters
    ----------
    path : Path to folder continaing individual worm images - the images used to generate each feature stack.
    the ground truths should be stored in a subfolder within this folder - "thresholded"

    Returns
    -------
    images - a list of image files
    gts - a list of ground truth files that should be in that subfolder
    path - a hand back of the path variable 

    '''
    allfiles = os.listdir(path) #List all files in folder
    images = [q for q in allfiles if q.endswith(".png")] #List comprehension to only take png files
    gtholder = images.copy() #A copy to let us mess with image names while keeping the list safe
    gts = [e[:-4]+"threshold.png" for e in gtholder] #List comprehesnion to make list of GTs **in same order as image list**.
    return(images, gts, path)#giveback images, gts, and path
#%%
def get_average_image_dimensions(images, path):
    '''
    

    Parameters
    ----------
    images : list of individual worm images
    path : path to individual images

    Returns
    -------
    average x and y, rounded UP to the nearest integer.

    '''
    xs = [] #Empty list for x coords
    ys = [] #Empty list for y coords
    for t in images: #t is an image in list
        image = cv2.imread(path + t) #Open 't' iteration
        x,y,z = image.shape #get image shape
        xs.append(x) #add x to x coords
        ys.append(y) #add y to y coords
    average_x = ceil(sum(xs)/len(xs)) #take average and ceiling it
    average_y = ceil(sum(ys)/len(ys)) #take average and ceiling it
    return(average_x, average_y) #return ceiling averages

    
def generate_features_concatinate_features(images, path):
    '''
    Parameters
    ----------
    images : list of image files - the files used to generate features
    path to image folder.

    Returns
    -------
    concatinated feature set for training RF
    '''
    dim = get_average_image_dimensions(images, path)
    feature_cat = np.empty((587, 582, 15))
    for r in range(0, len(images)):
        image = cv2.imread(path+images[r])
        gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gimage_uniform = cv2.resize(gimage, dim, interpolation=cv2.INTER_AREA)
        newfeatures = features_func(gimage_uniform)
        feature_cat = np.concatenate((feature_cat, newfeatures), 0)
    return(feature_cat)
sigma_max = 16
sigma_min = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)
def reshape_images_feature_gen_cat_out(path):
    images, gts, path = images_and_truths(path)
    feature_cat = generate_features_concatinate_features(images, path)
    return(feature_cat)
#%%
feature_cat = reshape_images_feature_gen_cat_out("D:/2022-10-24/correct_dy96/indi_worm/")
#%%
images, gts, path = images_and_truths("D:/2022-10-24/correct_dy96/indi_worm/")
average_x, average_y = get_average_image_dimensions(images, path)

#%%

sigma_max = 16
sigma_min = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)

def generate_features_concatinate_features(images, path):
    '''
    Parameters
    ----------
    images : list of image files - the files used to generate features
    path to image folder.

    Returns
    -------
    concatinated feature set for training RF
    '''
    feature_cat = np.empty(582, 587)
    for r in range(0, len(images)):
        image = cv2.imread(path+images[r])
        gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gimage_uniform = cv2.resize(image, (582, 587), interpolation=cv2.INTER_AREA)
        newfeatures = features_func(gimage_uniform)
        feature_cat = np.concatenate((feature_cat, newfeatures), 0)
        
#%%z = list of all features
feature_cat = np.append((features, features2), 0)
#%%z1 = np.concatenate((z[i] for i in range(0, len(z)), 1)



#%%

sigma_max = 16
sigma_min = 1

#I think I need to explicitly state the features - the new array has way less

features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)

#%%

features2 = features_func(gimage2)
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