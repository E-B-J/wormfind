# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:41:27 2022

@author: ebjam
"""
#Imports
import json
import math
import cv2

#%% Set up dir and lists
working_dir = "C:/Users/ebjam/Desktop/labeled/test/" # /leave/a/trailing/slash/
#%%Breaking down coco to individual images
def group_to_indie(workingjson):
    with open(workingjson) as test_json:
        test_set = json.load(test_json)
        sep_dict = {}
        for r in range(0, len(test_set["images"])):
        #extract image ID and name into a dictionary to be sure of ID later
            key = test_set["images"][r]['id']
            #imlist[test_set["images"][r]['file_name']] = test_set["images"][r]['id']
            temp_anno = {"annotations": []}
            sep_dict[test_set["images"][r]['file_name']] = {}
            for s in range(0, len(test_set["annotations"])):
                if test_set["annotations"][s]["image_id"] == key:!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    print("sorted annotation", s, "to image" + key)
                    temp_anno["annotations"].append(test_set["annotations"][s])
            sep_dict[test_set["images"][r]['file_name']] = temp_anno
            sep_dict[test_set["images"][r]['file_name']]["categories"] = test_set["categories"]
            sep_dict[test_set["images"][r]['file_name']]["images"] = test_set["images"][r]
    return(sep_dict)
#%%
modtest = group_to_indie("C:/Users/ebjam/Desktop/labeled/test.json")  

#%%
def bbox_relation(wormbbox, embryobbox):
    if wormbbox[0] < embryobbox[0] and wormbbox[1] < embryobbox[1]:
        if (wormbbox[0] + wormbbox[2]) > (embryobbox[0] + embryobbox[2]):
            if (wormbbox[1] + wormbbox[3]) > (embryobbox[1] + embryobbox[3]):
                inside = True
            else:
                inside = False
        else:
            inside = False
    else:
        inside = False
    return(inside)
#%%
def bbox_transpose(wormbbox, embryobbox):
    embryobbox[0] = embryobbox[0] - wormbbox[0]
    embryobbox[1] = embryobbox[1] - wormbbox[1]
    return(embryobbox)
#%%
def seg_transpose(wormbbox, embryo_seg):
    for y in range(0, len(embryo_seg[0])): #y is a single point within the segmentation coords
        embryo_seg[0][y] = embryo_seg[0][y] - wormbbox[y%2]
    return(embryo_seg)
#%%
working_dir = "C:/Users/ebjam/Desktop/labeled/test/"
transposed_embryo_out = {"annotations": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}], "images": []}
wormcount = 0
embryo_count = 0
hitcount=0
with open("C:/Users/ebjam/Desktop/labeled/test.json") as test_json:
    test_set = json.load(test_json)
    dump_dict = {"annotations": []}
    hitno = 0
    for q in range(0, len(test_set["images"])): #q is an image within the dataset
        im = cv2.imread(working_dir + test_set["images"][q]["file_name"])
        imid = test_set["images"][q]["id"]
        print(q)
        print(imid)
        imagewormlist = [] #make empty list of worms for this image
        imageemblist = [] #make empty list of embryos for this image
        super_giant_embryo_list = [i for i in test_set["annotations"] if i["category_id"] == 1]
        imagewormlist = [i for i in test_set["annotations"] if i["category_id"] == 2 and i["image_id"] == imid]
        for w in imagewormlist:
            one_worm_embryos = [i for i in super_giant_embryo_list if bbox_relation(w["bbox"], i["bbox"]) == True]
            for e in range(0, len(one_worm_embryos)):
                one_worm_embryos[e]["bbox"] = bbox_transpose(w["bbox"], one_worm_embryos[e]["bbox"])
                one_worm_embryos[e]["segmentation"] = seg_transpose(w["bbox"], one_worm_embryos[e]["segmentation"])
                one_worm_embryos[e]["id"] = hitno
                print("hit number:", hitno)
                hitno +=1 
                dump_dict["annotations"].append(one_worm_embryos[e])
            #%%
    dump_dict["categories"]
    dump_dict["images"]
            

        #%%
        #for w in range(0, len(test_set["annotations"]) if test_set["annotations"][w]["category_id"] == 1:
            #super_giant_embryo_list.append(test_set["annotations"][w])
            #for all annos 
            if test_set["annotations"][w]["category_id"] == 1:
                super_giant_embryo_list.append(test_set["annotations"][w])
            if test_set["annotations"][w]["image_id"] == imid: #if the annotation is within this image
                if test_set["annotations"][w]["category_id"] == 2: #If annotation is worm                         2 is worm, 1 is embryo, 0 is cluster
                    imagewormlist.append(test_set["annotations"][w]) #populate worm list
        for e in range(0, len(imagewormlist)):#e is a worm within an image
            worm_img_annotations = {"annotations": []} #make empty list of annotations witihn the worm bounding box
            single_worm_bbox = imagewormlist[e]["bbox"] #get worm bounding box
            print("wormbox", e)
            #Crop image to that worm bbox
            crop_image = im[math.floor(single_worm_bbox[1]):math.ceil(single_worm_bbox[1] + single_worm_bbox[3]), math.floor(single_worm_bbox[0]):math.ceil(single_worm_bbox[0]+single_worm_bbox[2])]
            cv2.imwrite(working_dir + test_set["images"][q]["file_name"][0:-4] + "_worm_" + str(e+1) + ".png", crop_image) #save cropped image
            keyname = test_set["images"][q]["file_name"][0:-4] + "_worm_" + str(e+1) + ".png" #setting the save name as a variable to store
            imdict = {} #empty image dictionary to be made for newly cropped image
            imdict["height"] = single_worm_bbox[3]
            imdict["width"] = single_worm_bbox[2]
            imdict["id"] = wormcount #id = count of worms from 0 - n-1
            imdict["file_name"] = keyname
            transposed_embryo_out["images"].append(imdict) #adding the image dictionary to the final JSON 
            for r in range(0, len(super_giant_embryo_list)):
                if super_giant_embryo_list[r]["image_id"] == imid: #r is an embryo within the image
                    holder = super_giant_embryo_list[r]
                    embryo_bbox = super_giant_embryo_list[r]["bbox"] #get embryo bbox
                    embryo_seg = super_giant_embryo_list[r]["segmentation"]
                    if embryo_bbox[0] => single_worm_bbox[0]: #if embryo minx > worm minx
                        if (embryo_bbox[0] + embryo_bbox[2]) =< (single_worm_bbox[0] + single_worm_bbox[2]): #if embryo maxx < worm maxx
                            if embryo_bbox[1] => single_worm_bbox[1]:#if embryo miny > worm miny
                                if (embryo_bbox[1] + embryo_bbox[3]) =< (single_worm_bbox[1] + single_worm_bbox[3]): #if embryo maxy < worm maxy
                                    #Embryo is inside worm bbox here!!
                                    print("hit #", hitcount)
                                    holder["id"] = hitcount
                                    hitcount+=1
                                    transposable_bbox = embryo_bbox
                                    transposable_bbox[0] = transposable_bbox[0] - single_worm_bbox[0] #transpose
                                    transposable_bbox[1] = transposable_bbox[1] - single_worm_bbox[1] #transpose
                                    holder["image_id"] = wormcount
                                    transposable_seg = embryo_seg
                                    for y in range(0, len(transposable_seg[0])): #y is a single point within the segmentation coords
                                        transposable_seg[0][y] = transposable_seg[0][y] - single_worm_bbox[y%2] #neat little guy to choose between x and y transposition
                                    holder["bbox"] = transposable_bbox
                                    holder["segmentation"] = transposable_seg
                                    transposed_embryo_out["annotations"].append(holder)
                                    #transposed_embryo_out["annotations"].append(worm_img_annotations["annotations"][r])
                                    #if embryo bbox is inside worm bbox - add it to worm_img_annotations
                                elif (embryo_bbox[1] + embryo_bbox[3]) => (single_worm_bbox[1] + single_worm_bbox[3])::
                                    
                            else:
                                break
                        else:
                            break
                    else:
                        break

                wormcount += 1 
    ''' 
           for t in range(0, len(worm_img_annotations["annotations"])): #t is an embryo annotation within the current worm
                transposable_bbox = worm_img_annotations["annotations"][t]["bbox"] #Get bbox from worm_img_list - temp file
                transposable_bbox[0] = transposable_bbox[0] - single_worm_bbox[0] #transpose
                transposable_bbox[1] = transposable_bbox[1] - single_worm_bbox[1] #transpose
                worm_img_annotations["annotations"][t]["bbox"] = transposable_bbox #put transpose back where it came from in temp file
                worm_img_annotations["annotations"][t]["image_id"] = wormcount
                transposable_seg = worm_img_annotations["annotations"][t]["segmentation"] #Get seg from temp
                for y in range(0, len(transposable_seg[0])): #y is a single point within the segmentation coords
                    transposable_seg[0][y] = transposable_seg[0][y] - single_worm_bbox[y%2] #neat little guy to choose between x and y transposition
                worm_img_annotations["annotations"][t]["segmentation"] = transposable_seg #put segmentation back where it came from
                transposed_embryo_out["annotations"].append(worm_img_annotations["annotations"][t]) #put the annotation from this loop file into the outfile
    '''
       
with open(working_dir + "worm_by_worm_embryo_test_transposed.json", "w") as outfile: #End of the "for j..." loop on line 59(or close to) - j iterates over worms
    json.dump(transposed_embryo_out, outfile, indent=4)