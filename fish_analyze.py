# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:07:55 2023

@author: ebjam
"""

import pickle, os, csv


start_folder = "C:/Users/ebjam/Downloads/2023-03-04/2023-03-04/ai/DY96/"

# Probably don't need to change full_image_pickle!
full_image_pickle = "auto_fish.pickle"
index_segmentation_record = os.path.join(start_folder, full_image_pickle)
file = open(index_segmentation_record,'rb')
seg_record = pickle.load(file)

#%%

threshimage = seg_record[0]['single_worms'][0]['fresh fish']

plt.imshow(threshimage)
#%%
res_dict = {}
for result in seg_record:
    input_image = result["input_image"]
    for single_worm in result["single_worms"]:
        worm = single_worm["wormID"]
        pa = single_worm["percent_infected"]
        res_dict[worm] = pa

#%%
import pandas as pd
df = pd.DataFrame(res_dict)

#%%
    
#CSV witer to catch everything and save - allows me to then add treat via excel
with open('C:/Users/ebjam/Downloads/2023-03-04/2023-03-04/ai/DY96/fish_out.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["image", "pa"])
    writer.writeheader()
    writer.writerows(res_dict)
#%%csv saver brought over from embryo and MS detector and stripped for testing/c3 specific use

def csv_saver(save_csv, res_dict, input_folder):
    if save_csv == 1:
        print("Saving results to csv within input/DY96 folder.")
        # Open csv to save stuff in
        savefile = open(os.path.join(input_folder,'fish_out.csv'), 'w', newline='')
        # Make writer write results to that save file
        writer = csv.writer(savefile)
        # Get column headers by getting list of by worm dictionary keys
        column_heads = list(.keys())
        # Write those headers to the first line of the file
        writer.writerow(column_heads)
        # Write values from each worm into csv
        for image in finaldict:
            for worm in image["single_worms"]:
                writer.writerow(worm.values())
        # Close file after writing
        savefile.close()
        print("Saving complete.")
    return()