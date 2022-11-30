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
with open("C:/Users/ebjam/Desktop/labeled/test.json") as test_json:
    test_set = json.load(test_json)
    dump_dict = {"annotations": [], "images": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}]}
    hitno = 0
    super_giant_embryo_list = [i for i in test_set["annotations"] if i["category_id"] == 1] # All embryos in every image
    for q in range(0, len(test_set["images"])): #for each image within the dataset
        im = cv2.imread(working_dir + test_set["images"][q]["file_name"]) #load the image with cv2
        imid = test_set["images"][q]["id"] #get the image ID to call all embryos and worms in the image
        print(q)
        print(imid)
        imagewormlist = [] #make empty list of worms for this image
        imageemblist = [] #make empty list of embryos for this image
        imagewormlist = [i for i in test_set["annotations"] if i["category_id"] == 2 and i["image_id"] == imid] #All worms just within an image
        for w in imagewormlist: #For each worm within the image
            one_worm_embryos = [i for i in super_giant_embryo_list if bbox_relation(w["bbox"], i["bbox"]) == True]
            crop_bbox = w["bbox"]
            cropim = im[math.floor(crop_bbox[1]):math.ceil(crop_bbox[1] + crop_bbox[3]), math.floor(crop_bbox[0]):math.ceil(crop_bbox[0]+crop_bbox[2])]
            cv2.imwrite(working_dir + test_set["images"][q]["file_name"][0:-4] + "_totalworm_" + str(wormcount) + ".png", cropim)
            keyname = test_set["images"][q]["file_name"][0:-4] + "_totalworm_" + str(wormcount) + ".png" #setting the save name as a variable to store
            imdict = {} #empty image dictionary to be made for newly cropped image
            imdict["height"] = crop_bbox[3]
            imdict["width"] = crop_bbox[2]
            imdict["id"] = wormcount #id = count of worms from 0 - n-1
            imdict["file_name"] = keyname
            dump_dict["images"].append(imdict)
            #print(one_worm_embryos)
            for e in range(0, len(one_worm_embryos)):
                temp_store = {}
                temp_store["bbox"] = bbox_transpose(w["bbox"], one_worm_embryos[e]["bbox"])
                #one_worm_embryos[e]["bbox"] = bbox_transpose(w["bbox"], one_worm_embryos[e]["bbox"])
                temp_store["segmentation"] = seg_transpose(w["bbox"], one_worm_embryos[e]["segmentation"])
                #one_worm_embryos[e]["segmentation"] = seg_transpose(w["bbox"], one_worm_embryos[e]["segmentation"])
                temp_store["id"] = hitno
                #one_worm_embryos[e]["id"] = hitno
                #print(one_worm_embryos[e]["id"])
                #one_worm_embryos[e]["image_id"] = wormcount
                temp_store["image_id"] = wormcount
                print("hit number:", hitno)
                hitno +=1 
                dump_dict["annotations"].append(temp_store)
            wormcount +=1
            
#%%
with open(working_dir + "worm_by_worm_embryo_test_transposed.json", "w") as outfile: #End of the "for j..." loop on line 59(or close to) - j iterates over worms
    json.dump(dump_dict, outfile, indent=4)            
#%%
dump_dict = {"annotations": [], "images": [], "categories": [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}]}
working_dir = "C:/Users/ebjam/Desktop/labeled/test/"
ann_to_image_dict = {"img0_annlist": [], "img1_annlist": [], "img2_annlist": [], "img3_annlist": [], "img4_annlist": [], "img5_annlist": []}
embryo_to_image_dict = {"img0_emblist": [], "img1_emblist": [], "img2_emblist": [], "img3_emblist": [], "img4_emblist": [], "img5_emblist": []}

wormcount = 0
embryocount = 0
with open("C:/Users/ebjam/Desktop/labeled/test.json") as test_json:
    test_set = json.load(test_json)
    super_giant_embryo_list = [i for i in test_set["annotations"] if i["category_id"] == 1]
    enumerated_embs = enumerate(super_giant_embryo_list)
    for ele in enumerated_embs:  
        #print(ele[1]["image_id"])
        if ele[1]["image_id"] == 0:
            embryo_to_image_dict["img0_annlist"].append(ele[1])
        if ele[1]["image_id"] == 1:
            embryo_to_image_dict["img1_annlist"].append(ele[1])
        if ele[1]["image_id"] == 2:
            embryo_to_image_dict["img2_annlist"].append(ele[1])
        if ele[1]["image_id"] == 3:
            embryo_to_image_dict["img3_annlist"].append(ele[1])
        if ele[1]["image_id"] == 4:
            embryo_to_image_dict["img4_annlist"].append(ele[1])
        if ele[1]["image_id"] == 5:
            embryo_to_image_dict["img5_annlist"].append(ele[1])
    super_giant_worm_list = [i for i in test_set["annotations"] if i["category_id"] == 2]
    enumerated_worms = enumerate(super_giant_worm_list)
    for ele in enumerated_worms:  
        #print(ele[1]["image_id"])
        if ele[1]["image_id"] == 0:
            embryo_to_image_dict["img0_emblist"].append(ele[1])
        if ele[1]["image_id"] == 1:
            embryo_to_image_dict["img1_emblist"].append(ele[1])
        if ele[1]["image_id"] == 2:
            embryo_to_image_dict["img2_emblist"].append(ele[1])
        if ele[1]["image_id"] == 3:
            embryo_to_image_dict["img3_emblist"].append(ele[1])
        if ele[1]["image_id"] == 4:
            embryo_to_image_dict["img4_emblist"].append(ele[1])
        if ele[1]["image_id"] == 5:
            embryo_to_image_dict["img5_emblist"].append(ele[1])
    for key in embryo_to_image_dict:
        current_image_set = embryo_to_image_dict[key]
        wormimgid = current_image_set[0]["image_id"]
        image_worm_list = [i for i in current_image_set if i["category_id"] == 2]
        #image_embryo_list = [i for i in current_image_set if i["category_id"] == 1]
        for image in test_set["images"]:
            if image["id"] == wormimgid:
                image_name = image["file_name"]
        im = cv2.imread(working_dir + image_name)
        for i in image_worm_list:
            
            single_worm_dict = {"annotations": []}
            single_worm_bbox = i["bbox"]
            crop_image = im[math.floor(single_worm_bbox[1]):math.ceil(single_worm_bbox[1] + single_worm_bbox[3]), math.floor(single_worm_bbox[0]):math.ceil(single_worm_bbox[0]+single_worm_bbox[2])]
            cv2.imwrite(working_dir + image_name + "_worm_" + str(wormcount) + ".png", crop_image)
            keyname = image_name + "_worm_" + str(wormcount) + ".png" #setting the save name as a variable to store
            imdict = {} #empty image dictionary to be made for newly cropped image
            imdict["height"] = single_worm_bbox[3]
            imdict["width"] = single_worm_bbox[2]
            imdict["id"] = wormcount #id = count of worms from 0 - n-1
            imdict["file_name"] = keyname
            dump_dict["images"].append(imdict)
            image_embryo_list = [i for i in current_image_set if i["category_id"] == 1]
            for embryo in image_embryo_list:
                embryobbox = embryo["bbox"]
                if bbox_relation(single_worm_bbox, embryobbox):
                    single_worm_dict["annotations"].append(embryo) ##UNTRANSPOSED + unrenamed
                    
            print(len(single_worm_dict["annotations"]))    
            for single_a in range(0, len(single_worm_dict["annotations"])):
                notran = single_worm_dict["annotations"][single_a]["bbox"]
                transposedbbox = bbox_transpose(single_worm_bbox, single_worm_dict["annotations"][single_a]["bbox"])
                transposed_segmentation = seg_transpose(single_worm_bbox, single_worm_dict["annotations"][single_a]["segmentation"])
                transpo_dict = single_worm_dict["annotations"][single_a]
                transpo_dict["bbox"] = transposedbbox
                transpo_dict["segmentation"] = transposed_segmentation
                transpo_dict["image_id"] = wormcount
                transpo_dict["id"] = embryocount
                dump_dict["annotations"].append(transpo_dict)
                embryocount+=1
                    
            wormcount+=1
            
    

#%%
img0list = embryo_to_image_dict["img0_emblist"]
        #%%
allstruc = {}
for ite in range(0, len(super_giant_worm_list)):
    worm = super_giant_worm_list[ite]
    wormlist = []
    for embryo in super_giant_embryo_list:
        if bbox_relation(worm["bbox"], embryo["bbox"]) == True:
            print(ite)
            print("hit")
            wormlist.append(embryo)
    allstruc["worm" + str(ite)] = wormlist
            #%%
with open(working_dir + "worm_by_worm_embryo_test_transposed.json", "w") as outfile: #End of the "for j..." loop on line 59(or close to) - j iterates over worms
    json.dump(dump_dict, outfile, indent=4)
            #%%
print(test_set["images"][q]["file_name"][0:-4] + "_totalworm_" + str(wormcount))      
            #%%
    dump_dict["categories"] = [{'supercategory': 'embryo', 'id': 1, 'name': 'embryo'}]
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