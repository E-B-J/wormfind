# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:17:44 2024

@author: ebjam
"""

import os
import pandas as pd

p1u = "D:/toronto_microscopy/ixmc/p1-U-nov14_Plate_2387/p1-U-nov14_Plate_2387/TimePoint_1/"

all_images = [q for q in os.listdir(p1u)]

alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

cols = ['01','02','03','04','05','06','07','08','09','10','11','12']

w_s = [1, 2, 3]

img_df = pd.DataFrame(all_images)
img_df['well'] = img_df[0].str[11:14]
#%%
wells = list(img_df['well'].unique())

for well in wells:
    this_well = img_df.loc[img_df['well'] == well].copy()
    if len(this_well) < 12:
        print(well)