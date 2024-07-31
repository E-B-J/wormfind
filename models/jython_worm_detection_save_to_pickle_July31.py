from ij import IJ, WindowManager, ImagePlus
from ij.plugin.frame import RoiManager
from ij.plugin import Duplicator
from ij.measure import ResultsTable
import os
import pickle
import gzip

# Fairly brute force worm detection: 
# Worms are ID'ed via the DAPI channel, forcing background and foreground via contrast and thresholding adjustments
# To seperate adjacent but non-ovrlapping worms I run erosion and sharpening, before hole filling to mend 'empty' signal in embryos.
# I then run a fairly unbias particle analysis (currently no shape filtering, and dropping particles larger than 30k as being overlaps)
# Each detection is then measured, and the segmentation recorded, before being pickled as a list of dictionaries.
# Each dictionary is a single detection, and contains the ROI name, the image name, and segmentation/location/description
# The easiest way to handle the output is probably to run pd.DataFrame(*your loaded pkl variable here*) to get tabular data.

# Major points for adjustment/customization: current lines 28(setMinAndMax), 29(setThreshold), and 32(erosion iterations)
# Questions? - Ask Ed. Or if he can't help, try ChatGPT.

# TODO: gzip the pickle (I promise that actually means something) - Might be done, run and check line 115

# Do particle counting etc as a function, then run on individual files within a loop - didn't have success running all together in one loop!
def getWorms(image):
	detected_worms = [] 
	#Duplicate image
	dup = Duplicator().run(image)
	# Automated contrast enhancement
	IJ.run(dup, "Enhance Local Contrast (CLAHE)", "blocksize=127 histogram=256 maximum=10 mask=*None* fast_(less_accurate)");
	# De-noise the contrast adjustment with a soft gaussian filter
	IJ.run(dup, "Gaussian Blur...", "sigma=2");
	# Divide foreground and background
	IJ.setMinAndMax(dup, 8000, 20000)
	IJ.setThreshold(dup, 15000, 65535)
    # Try to seperate worms
	IJ.run(dup, "Convert to Mask", "")
	IJ.run(dup, "Options...", "iterations=5 count=1 black do=Open")
	IJ.run(dup, "Options...", "iterations=3 count=2 black do=Erode")
	IJ.run(dup, "Sharpen", "")
	IJ.run(dup, "Sharpen", "")
	IJ.run(dup, "Sharpen", "")
	IJ.run(dup, "Find Edges", "")
	IJ.run(dup, "Fill Holes", "")
    # ID particles
	IJ.run(dup, "Analyze Particles...", "size=800-300000 circularity=0.00-1.00 show=Nothing add")
	roiManager = RoiManager.getInstance()
	if roiManager is None:
		roiManager = RoiManager()
	roiManager.deselect()
	numROIs = roiManager.getCount()
	# Close the duplicated image
	dup.close()
	# Force particle ROIs to original image to measure
	roiManager.runCommand("Show All")
	# Iterate over particles
	for q in range(numROIs):
		# Initiate detection and segmentation
		detection = {}
		segmentation = []
		tran_seg = []
		roiManager.select(q)
		roi = roiManager.getRoi(q)
		name = roi.getName()
		bounds = roi.getBounds()
		# Get BBOX as xywh - need to add these to detection to locate worm in DY96 channel
		x, y, width, height = bounds.x, bounds.y, bounds.width, bounds.height
		# Need segmentation, transposed and untransposed
		# First make untransposed, then subtract x and y to transpose to bbox
		roi_points = roi.getFloatPolygon()
		xpoints = roi_points.xpoints
		ypoints = roi_points.ypoints
		for w in range(len(xpoints)):
			point = (xpoints[w], ypoints[w])
			tran_point = (xpoints[w] - x, ypoints[w] - y)
			segmentation.append(point)
			tran_seg.append(tran_point)
		# Take measurements to filter on
		imp.setRoi(roi)
		IJ.run(imp, "Measure", "")
		rt = ResultsTable.getResultsTable()
		measurements = {}
		measurements['image'] = fileName # EXTRAPOLATE PLATE NAME HERE!
		measurements['roiName'] = name
		measurements['lookup'] = fileName + "_" + name
		measurements['segmentation'] = segmentation
		measurements['tran_seg'] = tran_seg
		if rt.getCounter() > 0:
			for column in rt.getHeadings():
				if column != 'Label':
					measurements[column] = rt.getValue(column, rt.getCounter() -1)
                detection = measurements
                rt.reset()
		detected_worms.append(detection)
	return(detected_worms)

dir = IJ.getDirectory("Select folder of DAPI onefields - right now have to run seperate for each plate")
fileList = os.listdir(dir)
IJ.run("Set Measurements...", "area mean modal min perimeter bounding shape feret's integrated median display redirect=None decimal=3 add")
all_detections = []
for fileName in fileList:
	if fileName.endswith(".TIF"):
		print(fileName)
		filePath = os.path.join(dir, fileName)
		imp = IJ.openImage(filePath)
		title = imp.getTitle()
		image_detections = getWorms(imp)
		# Clear ROIS here!!
		roiManager = RoiManager.getInstance()
		if roiManager is not None:
			roiManager.runCommand("Delete")
		else:
			print("ROI manager not open - could be no detections!")
		roiManager.reset()
		# Join image detections to all detections
		all_detections = all_detections + image_detections
		
# OUTSIDE OF THE LOOP!!!!
# Save outs
pickleName = IJ.getString("Enter pickle file name (no extension) for folder: " + dir, "unfiltered_worms")
if pickleName:
	savePath = os.path.join(dir, pickleName + ".gz")
	with gzip.open(savePath, "wb") as f:
		pickle.dump(all_detections, f)
	print("Wormies saved, good job")


		