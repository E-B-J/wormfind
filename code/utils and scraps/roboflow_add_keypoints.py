# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:52:34 2023

@authors:chatGPT  & Ed
prompt: 
     can you write python code to load a json of coco annotations, 
     and for each image loop through one specific class, 
     and convert it from a bounding box to a single point at the center of the bounding box, 
     and checks if it is inside the segmentation of another class of object? 
     If the centerpoint is inside the segmentation it should be added to the segmentation as a keypoint "pharynx"
    
Response:
import json
from shapely.geometry import Point, Polygon

# Load COCO annotations from JSON file
with open('annotations.json', 'r') as f:
    coco = json.load(f)

# Specify the class to loop through
class_name = 'car'

# Specify the class to check for intersection
check_class = 'road'

# Loop through images
for image in coco['images']:
    image_id = image['id']
    # Loop through instances of the class
    for annotation in coco['annotations']:
        if annotation['image_id'] == image_id:
            if annotation['category_id'] == coco['categories'].index(class_name):
                # Get bounding box coordinates
                x, y, width, height = annotation['bbox']

                # Convert to single point at center of bounding box
                center_x = x + width / 2
                center_y = y + height / 2
                center_point = Point(center_x, center_y)

                # Loop through the segmentation of another class
                for segmentation in coco['annotations']:
                    if segmentation['image_id'] == image_id:
                        if segmentation['category_id'] == coco['categories'].index(check_class):
                            # Create polygon from segmentation
                            polygon = Polygon(segmentation['segmentation'])

                            # Check if center point is inside polygon
                            if polygon.contains(center_point):
                                if 'keypoints' not in segmentation:
                                    segmentation['keypoints'] = []
                                segmentation['keypoints'].append({'pharynx': [center_x, center_y]})
                                print(f"The center point of the {class_name} is inside the {check_class} and added as keypoint")
                            else:
                                print(f"The center point of the {class_name} is not inside the {check_class}")

"""

import json
from shapely.geometry import Point, Polygon

def strike(text):
    return ''.join([u'\u0336{}'.format(c) for c in text])

# Create json save space
keypoint_annotations = {"images": [], "annotations": []}

# Load COCO annotations from JSON file
with open('C:/Users/ebjam/Downloads/WormID.v1i.coco/train/_annotations.coco.json', 'r') as f:
    coco = json.load(f)
# Specify the class to loop through
class_name = 'pharynx'

# Specify the class to check for intersection
check_class = 'worm'

# Loop through images
for image in coco['images']:
    image_id = image['id']
    # Limit future loops to annotations within the same image
    image_annotations = [annotation for annotation in coco['annotations'] if annotation['image_id'] == image_id]
    #For annotations within image
    pharynxes = []
    for annotation in image_annotations:
        #If annotation is pharynx (pharynx category is 2)
        if annotation['category_id'] == 2:
            # Get bounding box coordinates
            x, y, width, height = annotation['bbox']
            # Convert to single point at center of bounding box
            center_x = x + width / 2
            center_y = y + height / 2
            center_point = Point(center_x, center_y)
            pharynxes.append(center_point)
    # Loop through the segmentation of another class
    for segmentation in image_annotations:
        # Worm category is 3
        if segmentation['category_id'] == 3:
            # Create polygon from segmentation
            # First of all need to convert list of single coordinates to a list of points
            # Start with list of coords, x1, y1, x2, y2, .... xn, yn
            tuples_list = []
            # Iterate through the list in steps of 2
            for i in range(0, len(segmentation['segmentation'][0]), 2):
                # Add the current and next item to a tuple
                tuples_list.append((segmentation['segmentation'][0][i], segmentation['segmentation'][0][i+1]))
            polygon = Polygon(tuples_list)
            for center_point in pharynxes:
                if polygon.contains(center_point):
                    if 'keypoints' not in segmentation:
                        # Make keypoint list
                        segmentation['keypoints'] = []
                        # Add segmentation to keypoint list, have to add 'v' coord (x, y, v):
                        # 0 = not labelled, 1 = labeled not visable, 2 = labelled and visible
                        segmentation['keypoints'].append([center_x, center_y, 2])
                        # Add modified segmentation to the segmentaiton list of save object
                        keypoint_annotations["annotations"].append(segmentation)
                        print("Pharynx matched to worm")
                    else:
                        print("Nope 1 - Annotation already has keypoints")
                else:
                    print("Nope 2 - Thank you Mario! But our " + strike("Princess") + " pharynx is in another " + strike("castle") + " worm")
    keypoint_annotations["images"].append(image)
keypoint_annotations["categories"] = coco["categories"]
#Add pharynx to keypoint of worms. Worms are category #3
keypoint_annotations["categories"][3]["keypoints"] = ["pharynx"]
keypoint_annotations["licenses"] = coco["licenses"]
keypoint_annotations["info"] = coco["info"]
    
#%%
with open("C:/Users/ebjam/Documents/GitHub/wormfind/keypoint_train_coco_annotations.json", "w") as json_file:
    json.dump(keypoint_annotations, json_file)