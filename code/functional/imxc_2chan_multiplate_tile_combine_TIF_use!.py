# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 08:26:30 2023

@author: ebjames

"""
#%%

import cv2, os
from PIL import Image

# Location/channel handling - input whatever the path is with all images from the plate scan
base_path = "E:/Toronto_microscopy/July-9-n2-worm-curve-lot19_Plate_2101/"

plates = [q for q in os.listdir(base_path) if not q.endswith('zip')]
plates = [q for q in plates if not q.endswith('csv')]
plates = [q for q in plates if not q.endswith('py')]
plates = [q for q in plates if not q.endswith('xlsx')]
#%%
for plate in plates:
    plate_path = base_path + plate + "/TimePoint_1/"
    # Assuming 2 channels = add more as needed
    dapi_path = plate_path + "dapi/"
    dy96_path = plate_path + "dy96/"
    #tl_path = plate_path + "tl/"
    chan_paths = [dapi_path, dy96_path]
    

# Make subfolders - channel by channel and a sub one_field folder with the combo image
    for path in chan_paths:    
        if not os.path.isdir(path):
            os.mkdir(path)
        one_field_path = path + 'one_field/'
        if not os.path.isdir(one_field_path):
            os.mkdir(one_field_path)

    # Getting a list of images for each wavelength. Wavelength is shown by number following 'w' in the tile title
    all_imgs = [q for q in os.listdir(plate_path) if q.endswith('TIF')]

    # -4 to deal with an error I made a round ago and didn't notice!
    w1_imgs = [w for w in all_imgs if w[-5] == '1']
    w2_imgs = [w for w in all_imgs if w[-5] == '2']
    #w3_imgs = [w for w in all_imgs if w[-5] == '3']
    def convert_to_png(w_list, chan_path):
        for infile in w_list:
            read = cv2.imread(plate_path + infile, -1)
            outfile = infile[:-3]+'TIF'
            cv2.imwrite(chan_path+outfile,read)

    convert_to_png(w1_imgs, dapi_path)
    convert_to_png(w2_imgs, dy96_path)

    #get all w1s1 image - using top left corner to build the rest
    w1_s1_images = [r[:-3] + 'TIF' for r in w1_imgs if r[-8] == '1']

    for top_left in w1_s1_images:
        tr = top_left[:-8] + '2' + top_left[-7:]
        bl = top_left[:-8] + '3' + top_left[-7:]
        br = top_left[:-8] + '4' + top_left[-7:]
        #s1 = Image.open(dapi_path + top_left)
        s1 = cv2.imread(os.path.join(dapi_path + top_left), -1)
        #s2 = Image.open(dapi_path + tr)
        s2 = cv2.imread(os.path.join(dapi_path + tr), -1)
        #s3 = Image.open(dapi_path + bl)
        s3 = cv2.imread(os.path.join(dapi_path + bl), -1)
        #s4 = Image.open(dapi_path + br)
        s4 = cv2.imread(os.path.join(dapi_path + br), -1)
        # Get the size of the original image
        top_row = cv2.hconcat([s1, s2])
        bottom_row = cv2.hconcat([s3, s4])
        combined_image = cv2.vconcat([top_row, bottom_row])
        #width, height = s1.size
        # Calculate the size of the combined image
        #combined_width = width * 2
        #combined_height = height * 2
        
        # Create a new image with the calculated size
        #combined_image = Image.new("RGB", (combined_width, combined_height))
        
        # Paste the tiles into the combined image
        #combined_image.paste(s1, (0, 0))
        #combined_image.paste(s2, (width, 0))
        #combined_image.paste(s3, (0, height))
        #combined_image.paste(s4, (width, height))
        
        # Save the combined image
        #combined_image.save(dapi_path + 'one_field/' + top_left[:-10] + 'w1_combo.png')
        cv2.imwrite(dapi_path + 'one_field/' + top_left[:-10] + 'w1_combo.TIF', combined_image)
    # Same deal on w2 as w1 - if we have a w3 I'll write this into a function to make it nicer to look at!
    w2_s1_images = [r[:-3] + 'TIF' for r in w2_imgs if r[-8] == '1']
    
    for top_left in w2_s1_images:
        tr = top_left[:-8] + '2' + top_left[-7:]
        bl = top_left[:-8] + '3' + top_left[-7:]
        br = top_left[:-8] + '4' + top_left[-7:]
        #s1 = Image.open(dy96_path + top_left)
        s1 = cv2.imread(os.path.join(dy96_path + top_left), -1)
        #s2 = Image.open(dy96_path + tr)
        s2 = cv2.imread(os.path.join(dy96_path + tr), -1)
        #s3 = Image.open(dy96_path + bl)
        s3 = cv2.imread(os.path.join(dy96_path + bl), -1)
        #s4 = Image.open(dy96_path + br)
        s4 = cv2.imread(os.path.join(dy96_path + br), -1)
    
        top_row = cv2.hconcat([s1, s2])
        bottom_row = cv2.hconcat([s3, s4])
        combined_image = cv2.vconcat([top_row, bottom_row])
        cv2.imwrite(dy96_path + 'one_field/' + top_left[:-10] + 'w1_combo.TIF', combined_image)
        # Get the size of the original image
        #width, height = s1.size
        # Calculate the size of the combined image
        #combined_width = width * 2
        #combined_height = height * 2
        
        # Create a new image with the calculated size
        #combined_image = Image.new("RGB", (combined_width, combined_height))
        
        # Paste the tiles into the combined image
        #combined_image.paste(s1, (0, 0))
        #combined_image.paste(s2, (width, 0))
        #combined_image.paste(s3, (0, height))
        #combined_image.paste(s4, (width, height))
        
        # Save the combined image
        #combined_image.save(dy96_path + 'one_field/' + top_left[:-10] + 'w2_combo.png')
    
'''    
w3_s1_images = [r[:-3] + 'png' for r in w3_imgs if r[-8] == '1']

for top_left in w3_s1_images:
    tr = top_left[:-8] + '2' + top_left[-7:]
    bl = top_left[:-8] + '3' + top_left[-7:]
    br = top_left[:-8] + '4' + top_left[-7:]
    s1 = Image.open(tl_path + top_left)
    s2 = Image.open(tl_path + tr)
    s3 = Image.open(tl_path + bl)
    s4 = Image.open(tl_path + br)

    # Get the size of the original image
    width, height = s1.size
    # Calculate the size of the combined image
    combined_width = width * 2
    combined_height = height * 2
    
    # Create a new image with the calculated size
    combined_image = Image.new("RGB", (combined_width, combined_height))
    
    # Paste the tiles into the combined image
    combined_image.paste(s1, (0, 0))
    combined_image.paste(s2, (width, 0))
    combined_image.paste(s3, (0, height))
    combined_image.paste(s4, (width, height))
    
    # Save the combined image
    combined_image.save(tl_path + 'one_field/' + top_left[:-10] + 'w3_combo.png')
'''