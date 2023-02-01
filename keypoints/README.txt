Keypoints made using bounding boxes in ROBOFLOW - then converted into keypoint using python to check if center point of pharynx bbox is inside worm segmentation

Augmentations done in ROBOFLOW:
	- resized to 688x552 - 1/4 size maintaining aspect ratio
	- bbox blur - 10px
	- sheer +- 15o - consider removing due to clustered worms getting broken up

Used '4x' in roboflow to make image sets of original image + 3x random augmented versions