# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:42:00 2022

@author: ebjam
"""
#%% Imports
import os #accessing/handling files
import numpy as np #image and data handling
import matplotlib.pyplot as plt #showing images/stats
from skimage import feature, future #RF tools
from sklearn.ensemble import RandomForestClassifier # The RF itself
from sklearn.model_selection import train_test_split #Split data
from sklearn.metrics import accuracy_score, classification_report #Assess model accuracy past oob
from functools import partial #For feature function
from math import ceil #Rounding up for pixel coordinates
import cv2 #Image handling
import joblib #Saving/loading model
import matplotlib.lines as mlines #Visualization util
import random
import PIL

#%%Functions
'''
def test_train_division(path):
    """
    selects and moves test and train images to folders.

    Parameters
    ----------
    path : path to folder with all individual worms in.

    Returns
    -------
    test_path: the path to the test folder
    train_path: path to the train folder

    """
    all_files = os.listdir(path)
    images = [q for q in all_files if q.endswith(".png")] #Only take png files
    print(len(images))
    train_draw=15
    trainset = random.sample(images, k=train_draw)
    print("Splitting off trainset of ", train_draw, " images.")
    for image in trainset:
        os.rename(path + image, path + "train/" + image)
        gt = path + "thresholded/" + image[:-4] + "threshold.png"
        os.rename(gt, path + "train/thresholded/" + image[:-4] + "threshold.png")
    print("Moved trainset to subfolder.")
    all_files_minus_train = os.listdir(path)
    images_minus_train = [w for w in all_files_minus_train if w.endswith(".png")]
    testset = random.sample(images_minus_train, k=train_draw)
    for image in testset:
        os.rename(path + image, path + "train/" + image)
        gt = path + "thresholded/" + image[:-4] + "threshold.png"
        os.rename(gt, path + "train/thresholded/" + image[:-4] + "threshold.png")
    print(len(images_minus_train))
    testpath = path + "test/"
    trainpath = path + "train/"
    return(trainpath, testpath)

#%%

a, b = test_train_division("D:/2022-10-24/correct_dy96/indi_worm/gravid/")
'''
#%%
def multic_to_onec(image):
    #Get size of image
    w, h = image.size
    #make array of zeros the size of input image
    one_channel_rescale = np.empty((w, h))
    #convert input image to pixels
    pix = image.load()
    #iterate through input image, read background/embryos/microsporidia
    for i in range(0,image.size[0]):
        for j in range(0, image.size[1]):
            pixel_value = pix[i, j]
            # Background = 1
            if pixel_value == (0,0,0):
                one_channel_rescale[i][j] = 0
            # Microsporidia = 3
            elif pixel_value == (0,0,255):
                one_channel_rescale[i][j] = 1
            #Background = 1
            else:
                one_channel_rescale[i][j] = 0.5
    return(one_channel_rescale)
#%%


def images_and_truths(path):
    '''
    Parameters
    ----------
    path : Path to folder continaing individual worm images - the images used
    to generate each feature stack.
    
    The ground truths should be stored in a subfolder within this folder - "thresholded"

    Returns
    -------
    images - a list of image files
    gts - a list of ground truth files that should be in that subfolder
    path - a hand back of the path variable 
    '''
    allfiles = os.listdir(path) #List all files in folder
    images = [q for q in allfiles if q.endswith(".png")] #Only take png files
    return(images)#giveback images, gts, and path

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
#%%
sigma_max = 16
sigma_min = 1
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max)
#%%
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
    feature_cat = np.empty((dim[1], dim[0], 20)) #!!!
    traininglabels = np.zeros((dim[1], dim[0]))
    for r in range(0, len(images)):
        image = cv2.imread(path+images[r])
        gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gimage_uniform = cv2.resize(gimage, dim, interpolation=cv2.INTER_AREA) #!!!!!!!!
        newfeatures = features_func(gimage)
        feature_cat = np.concatenate((feature_cat, newfeatures), 0)
        traininglabel = PIL.Image.open(path + "thresholded/"+images[r])
        one_channel_tl_uniform = multic_to_onec(traininglabel)
        #Deal with making it one channel here!!
        traininglabels = np.concatenate((traininglabels, one_channel_tl_uniform), 0)
        print("Image " + str(r+1) + " complete.")
    return(feature_cat, traininglabels)

def reshape_images_feature_gen_cat_out(path):
    '''
    Parameters
    ----------
    path :  Path to folder continaing individual worm images - the images used
    to generate each feature stack.
    The ground truths should be stored in a subfolder within this folder - "thresholded"

    Returns
    -------
    feature_cat :  Concatenated features
    gts :  Concatenated ground truths
    '''
    images = images_and_truths(path)
    feature_cat, gts = generate_features_concatinate_features_gts(images, path)
    return(feature_cat, gts)
#%%Random Forest

#Generate concatenated features and groundtruths
feature_cat, gts = reshape_images_feature_gen_cat_out("C:/Users/ebjam/Documents/GitHub/wormfind/rf/trinary_rf/")

#%%

#Groundtruths were loaded as multichannel - split out one channel
b,g,r = cv2.split(gts)

#%%
#Delete arrays we don't need to free up some memory
del g
del r
del gts

'''
Convert training labels to be a little more RF friendly

Class 1 is background
Class 2 is microsporidia

'''
b[b == 0] = 1#!!!
b[b == 255] = 2#!!!
#%%
#Make classifier
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                             max_depth=10)
#%%
#Train classifier - full fat, all features.
clf = future.fit_segmenter(b, feature_cat, clf)

#%%


#%%
'''
intensities = []
sobel_edges = []
structures = []
#%%
current_clf = joblib.load("C:/Users/ebjam/Documents/GitHub/wormfind/model02_20230113_100_trees_20_branches.joblib")

#%%
l = len(current_clf.feature_importances_)
intensities.append(current_clf.feature_importances_[:l//3]) #Intensity features
sobel_edges.append(current_clf.feature_importances_[l//3:2*l//3]) # Sobel edge addition
structures.append(current_clf.feature_importances_[2*l//3:]) # Structure features

#%%
del(current_clf)
#%%
#%%
y_pred_test = clf.predict(model1_feat_test)
accuracy_score(model1_feat_test, y_pred_test)
#%%
print(classification_report(model1_feat_test, y_pred_test))
#%%
#Plot feature importance
l = len(clf.feature_importances_)
feature_importance = (
        clf.feature_importances_[:l//3], #intensity
        clf.feature_importances_[l//3:2*l//3], #Sobel edge
        clf.feature_importances_[2*l//3:]) # Structure
#%%
sigmas = np.logspace(
        np.log2(sigma_min), np.log2(sigma_max),
        num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
        base=2, endpoint=True)

fig, ax = plt.subplots(1, 2, figsize=(9, 4))

#Intensity features
ax[0].plot(sigmas, feature_importance[0], 'D', color = 'k', label = "Intensity")
ax[0].set_title("Intensity features")
ax[0].set_xlabel("$\\sigma$") #Character call for sigma
ax[0].set_ylabel("Feature Importance")

#Structural features
ax[1].plot(sigmas, feature_importance[1], 'o', color = 'b', label = "Edges via gradient intensity")
ax[1].plot(sigmas, feature_importance[2], 's', color = 'r', label = "Structure")
ax[1].set_title("Texture features")
ax[1].set_xlabel("$\\sigma$")
ax[1].set_ylabel("Feature Importance")

#Plot horizontal line showing the mean importance (1/#features)
mean_importance = sum(clf.feature_importances_)/l #(always going to be 1 / l but good to check?)
ax[0].axhline(y=mean_importance, color='darkgrey', linestyle='--')
ax[1].axhline(y=mean_importance, color='darkgrey', linestyle='--')

#Legend construction
D = mlines.Line2D([], [], color='k', marker='D', linestyle='None',
                          markersize=10, label='Intensity')
o = mlines.Line2D([], [], color='b', marker='o', linestyle='None',
                          markersize=10, label='Intensity Gradient')
s = mlines.Line2D([], [], color='r', marker='s', linestyle='None',
                          markersize=10, label='Structure')
mean_importance = mlines.Line2D([], [], color='darkgrey', marker = 'D', linestyle='--',
                          markersize=0, label='Mean Feature importance')
fig.legend(loc = "upper center",handles=[D, o, s, mean_importance], bbox_to_anchor=(0.5, 1.04), ncol=4, fancybox=True)
#%%Save model
joblib.dump(clf, "C:/Users/ebjam/Documents/GitHub/wormfind/model02_20230113_100_trees_20_branches.joblib")

