directory = getDir("Choose image directory");
outdir = getDir("Pick output directory");
setBatchMode(true);
filelist = getFileList(directory);
for (i = 0; i < lengthOf(filelist); i++) {
    if (endsWith(filelist[i], ".czi")) { 
        //open(directory + File.separator + filelist[i]);
        run("Bio-Formats Importer", "open=" + directory + filelist[i] + " autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
        //run("Export HDF5", "select="+outdir+filelist[i][0:-3]".h5 exportpath="+outdir+filelist[i][0:-3]+".h5 datasetname=data compressionlevel=0 input="+filelist[i]);
        print(filelist[i]);
		titlesave = filelist[i].replace("czi", "h5");
        run("Save to HDF5 File (new or replace)...", "save=D:/2022-11-15/h5/"+titlesave);
        close();
    } 
}
