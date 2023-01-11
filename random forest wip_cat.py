# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:42:00 2022

@author: ebjam

worm by worm folder = D:/2022-10-24/correct_dy96/indi_worm/
threshold folder = D:/2022-10-24/correct_dy96/indi_worm/thresholded/
example worm: "D:/2022-10-24/correct_dy96/indi_worm/72hrn2i_25u_worm_1.png"
example gt: "D:/2022-10-24/correct_dy96/indi_worm/thresholded/72hrn2i_25u_worm_1threshold.png"
"""
#%% Imports
import os #accessing files
import numpy as np #image and number handling
import matplotlib.pyplot as plt #showing images
from skimage import data, segmentation, feature, future #RF tools
from sklearn.ensemble import RandomForestClassifier # RF
from sklearn.model_selection import train_test_split
from functools import partial #Feature gen
from math import ceil
import cv2 #Image handling
import joblib #Saving/loading model
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
#%%
path = "D:/2022-10-24/correct_dy96/indi_worm/gravid/threshold/"
allfiles = os.listdir(path)

rename_files = [q for q in allfiles if q.endswith(".pngthreshold.png")]

for w in range(0, len(rename_files)):
    os.rename(path + rename_files[w], path + rename_files[w][:-17]+"threshold.png")

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

sigma_max = 16
sigma_min = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)

def generate_features_concatinate_features_gts(images, path):
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
    print(dim)
    feature_cat = np.empty((dim[1], dim[0], 15))
    traininglabels = np.zeros((dim[1], dim[0], 3))
    for r in range(0, len(images)):
        image = cv2.imread(path+images[r])
        gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gimage_uniform = cv2.resize(gimage, dim, interpolation=cv2.INTER_AREA)
        newfeatures = features_func(gimage_uniform)
        feature_cat = np.concatenate((feature_cat, newfeatures), 0)
        traininglabel = cv2.imread(path + "threshold/"+images[r][:-4] + "threshold.png")
        tl_uniform = cv2.resize(traininglabel, dim, interpolation = cv2.INTER_AREA)
        traininglabels = np.concatenate((traininglabels, tl_uniform), 0)
    return(feature_cat, traininglabels)



def reshape_images_feature_gen_cat_out(path):
    images, gts, path = images_and_truths(path)
    feature_cat, gts = generate_features_concatinate_features_gts(images, path)
    return(feature_cat, gts)
#%%
feature_cat, gts = reshape_images_feature_gen_cat_out("D:/2022-10-24/correct_dy96/indi_worm/gravid/")

b,g,r = cv2.split(gts)
del g
del r
del gts

b[b == 0] = 1
b[b == 255] = 2

feature_train, feature_test, gts_train, gts_test = train_test_split(feature_cat, b, test_size=0.8, random_state=5320)
del feature_cat
del b
del feature_test
del gts_test

#%%
model1_feat_train, model1_feat_test, model1_label_train, model1_label_test = train_test_split(feature_train, gts_train, test_size = 0.5)
#%%
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                             max_depth=10, max_samples=0.05)
#%%
clf = future.fit_segmenter(model1_label_train, model1_feat_train, clf)

#%%
l = len(clf.feature_importances_)
feature_importance = (
        clf.feature_importances_[:l//3],
        clf.feature_importances_[l//3:2*l//3],
        clf.feature_importances_[2*l//3:])

sigmas = np.logspace(
        np.log2(sigma_min), np.log2(sigma_max),
        num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
        base=2, endpoint=True)

fig, ax = plt.subplots(1, 2, figsize=(9, 4))


ax[0].plot(sigmas, feature_importance[0], 'D', color = 'k', label = "Intensity")
ax[0].set_title("Intensity features")
ax[0].set_xlabel("$\\sigma$")
ax[0].set_ylabel("Feature Importance")

ax[1].plot(sigmas, feature_importance[1], 'o', color = 'b', label = "Edges via gradient intensity")
ax[1].plot(sigmas, feature_importance[2], 's', color = 'r', label = "Structure")
ax[1].set_title("Texture features")
ax[1].set_xlabel("$\\sigma$")
ax[1].set_ylabel("Feature Importance")

mean_importance = sum(clf.feature_importances_)/l #(always going to be 1 / l but good to check?)
ax[0].axhline(y=mean_importance, color='darkgrey', linestyle='--')
ax[1].axhline(y=mean_importance, color='darkgrey', linestyle='--')

D = mlines.Line2D([], [], color='k', marker='D', linestyle='None',
                          markersize=10, label='Intensity')
o = mlines.Line2D([], [], color='b', marker='o', linestyle='None',
                          markersize=10, label='Intensity Gradient')
s = mlines.Line2D([], [], color='r', marker='s', linestyle='None',
                          markersize=10, label='Structure')
mean_importance = mlines.Line2D([], [], color='darkgrey', marker = 'D', linestyle='--',
                          markersize=0, label='Mean Feature importance')

fig.legend(loc = "upper center",handles=[D, o, s, mean_importance], bbox_to_anchor=(0.5, 1.04), ncol=4, fancybox=True)

#%% From feature importance plot - probably dont need sigma 2, 4, 16 in intensity. Might not need intensity gradient 1, 4. Might not need structure 1, 8
#%%
ax[0].plot(sigmas, feature_importance[::3], 'o'),
ax[0].set_title("Intensity features")
ax[0].set_xlabel("$\\sigma$")
#%%
joblib.dump(clf, "C:/Users/ebjam/Documents/GitHub/wormfind/model02_20230104.joblib")
#%%
result = future.predict_segmenter(feature_test, clf)

#%%
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 4))
ax[0].imshow(segmentation.mark_boundaries(img_new, result_new, mode='thick'))
ax[0].set_title('Image')
ax[1].imshow(result_new)
ax[1].set_title('Segmentation')
fig.tight_layout()

plt.show()