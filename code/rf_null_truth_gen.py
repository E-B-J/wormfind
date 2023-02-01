# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:27:08 2022

@author: ebjam
"""

import cv2
from os import listdir, rename
import numpy as np
#%%
uni_dir = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/done/problems/"

uniworms = listdir(uni_dir)

for i in uniworms:
    img = cv2.imread(uni_dir + i)
    h, w, d = img.shape
    gt = np.zeros((h, w))
    cv2.imwrite(uni_dir + i[:-4] + "threshold.png", gt)


#%%
alldir = "C:/Users/ebjam/Documents/GitHub/wormfind/rf/done/"
alldirlist = listdir(alldir)
#%%
misname =[i for i in alldirlist if i[-17] == '.']
#%%
for filename in misname:
    rename(alldir + filename, alldir + filename[:-17] + 'threshold.png')