path = getDirectory("Choose image dir - czis");
jsonpath = getDirectory("Choose json dir");
list_of_files = getFileList(path);
num_files = list_of_files.length;
alljsons = getFileList(jsonpath);
numalljsons = jsons.length;
jsons = newArray(0);
for (h=0;h<numalljsons;h++)
	{
	json = alljsons[h];
		if (endsWith(file, "json"))
		{
			jsons.push(json)
		}
	}
for (g=0;g<jsons.length;g++)
	{
		currentjson = File.openAsString(jsonpath + jsons[g])
		
	}
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!jsons = load those jsons!
for(i=0; i<num_files;i++)
	{
		file = list_of_files[i];
		if (endsWith(file, "czi")){
			run("Bio-Formats", "open=[" + path + file +"] autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");
			search_handle = file.substring(0, str.length - 4);
			imgcoords = newArray(0);
			for (j=0; j < numjsons; j++) //Just the 48hr ones
			{
				json = jsons[j]
				segmentation = json.segmentation;
					if segmentation.id == search_handle;
						imgcoords.push(segmentation.segmentation);
			}
			for(k=0, k < imgcoords.length; k++)
			{
				seg = imgcoords[k];
				imgseg = newArray(0);
				no_points = (seg.length/2);
				for(l=0; l < no_points; i++)
					{
					
					imgseg.push(seg[l]);
					imgseg.push(seg[no_points + 1 + l]);
					}
				makePolygon(imgseg);
				roiManager("Add");
		}
	}