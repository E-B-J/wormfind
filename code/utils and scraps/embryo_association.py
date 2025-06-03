# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:54:18 2024

@author: ebjam

Assign embryos with worms, DO once, and then do confidence filtering post
"""

import joblib, cv2
import pandas as pd



l4_v_cyc1 = joblib.load(r"E:\toronto_microscopy\ixmc\sep_30_success\N2_lv4440_cyc1_curve\N2 lv4440 vs cyc1 2mil_Plate_2281\TimePoint_1\dapi\one_field\nov15_results.joblib")
plate = pd.DataFrame(l4_v_cyc1)
embryo_data = joblib.load(r"E:\toronto_microscopy\ixmc\sep_30_success\N2_lv4440_cyc1_curve\n2_cyc1_all_image_embryo_preds_all_conf_dec3.joblib")

plate['well'] = plate['image'].str[-15:-12]
plate['row'] = plate['well'].str[0]
plate['col'] = plate['well'].str[1:]
cyc1_rows = ['E', 'F', 'G', 'H']
spore_max = 2000000
def gen_spore_dict(spore_max):
    spore_dict = {}
    for i in range(0, 11):
        plate_col = str(i+1)
        spores = spore_max / 2**i
        if len(plate_col) <2:
            plate_col = '0' + plate_col
        spore_dict.update({plate_col: spores})
    spore_dict.update({'12' : 0})
    return(spore_dict)
di = gen_spore_dict(spore_max)

plate['#spore'] = plate['col'].map(di)

plate['bacterial_line'] = "L4440"
plate['bacterial_line'] = plate['bacterial_line'].where(~plate['row'].isin(cyc1_rows), 'cyc1')

worms = plate.loc[plate['worm_prediction'] == 'good'].copy()

# The rest iof filtering!!
worms['fish_5k_percent'] = worms['fish_pixels_5k'] / worms['mask_area']
worms['dy96_10k_percent'] = worms['dy96_pixels_10k'] / worms['mask_area']
worms['spores_per_ul'] = round(worms['#spore'] / 65, 2)
reps = [1, 2, 3]

#Maybe go ahead and filter thos >95% detections to remove what is almost certainly noise....
ironsii = worms.loc[worms['fish_5k_percent'] < 75].copy()
ironsii = ironsii.loc[ironsii['dy96_percent_10k'] < 60].copy()
# And the detections below 3750 are probabl debris
ironsii = ironsii.loc[ironsii['area'] > 3750].copy()
ironsii = ironsii.loc[ironsii['area'] < 30000].copy()
worms = ironsii.copy()


embo = pd.DataFrame(embryo_data)

def embryo_search(image_name, embryo_data):
    hits = [q for q in embryo_data if q['image'] == image_name]
    return(hits)
#%%
worms['internal_embryos'] = [[] for _ in range(worms.shape[0])] #Make empty list
for image in worms['image'].unique():
    dy_image = image.replace('w1_combo', 'w2_combo')
    this_image_embryos = embryo_search(dy_image, embryo_data)
    if len(this_image_embryos) > 0:
        embryos = this_image_embryos[0]['embryos']
    else:
        continue

    for i, row in worms.loc[worms['image'] == image].iterrows():
        this_contour = row['contour']
        internal_embryos = row['internal_embryos']
        for embryo in embryos:
            if cv2.pointPolygonTest(this_contour, embryo['centroid'], False) != -1: # If embryo is outside contour, not on or in
                internal_embryos.append(embryo)
        worms.at[i, 'internal_embryos'] = internal_embryos
          

    
# I'd like a difference between didn't run and didn't contain embryos
#%%
