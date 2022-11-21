setBatchMode(true);
path = getDirectory("Choose image dir");
polypath = getDirectory("Choose poly/seg/csv path");
chan_no = getNumber("What channel are we thresholding? Usually DY96 = 2, red FISH = 3", 3);
list_of_files = getFileList(path);
num_files = list_of_files.length;
list_of_poly = getFileList(polypath);
for(czifile=0; czifile<num_files; czifile++){
	file = list_of_files[czifile];
	if (endsWith(file, "czi")){
			run("Bio-Formats Importer", "open="+path+file +" autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
			open(path+file);
			t = getTitle();
			anno_title = t.replace(".czi", ".csv");
			open(polypath + anno_title);
			no_polygons = Table.size;
			noarray=newArray(0);
			xpoints=newArray(0);
			ypoints=newArray(0);
			for(k=0; k<no_polygons;k++){
				singlepoly = Table.getString("seg", k);
				sp_array = split(singlepoly, ",");
				for(a=0; a<sp_array.length;a++){
					floatval = parseInt(sp_array[a]);
					noarray = Array.concat(noarray, sp_array[a]);
					if (a % 2 == 0){
						xpoints = Array.concat(xpoints, sp_array[a]);
					}
					else{
						ypoints = Array.concat(ypoints, sp_array[a]);
				//makeSelection(“polygon”, xpoints, ypoints);
				//this is where the problem happens....
				makePolygon(noarray);
				roiManager("add");
				}
			}
			//Now, threshold the proper channel!
			Stack.setChannel(chan_no);
			run("Set Measurements...", "area min area_fraction display redirect=None decimal=3");
			setAutoThreshold("Default dark no-reset");
			//run("Threshold...");
			setThreshold(2100, 65535, "raw");	
			roiManager("Deselect");
			roiManager("Measure");
			roiManager("Deselect");
			roiManager("Delete");
close();
	}
}