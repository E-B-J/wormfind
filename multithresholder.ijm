path = getDirectory("Choose image dir");
list_of_files = getFileList(path);
num_files = list_of_files.length;
for(n=0; n<num_files;n++)
	{
		file = list_of_files[n];
		open(path+file);
		run("8-bit");
		run("Threshold...");
		waitForUser("Press ok after thresholding.");
		setOption("BlackBackground", true);
		run("Convert to Mask");
		savetitle = filename.replace(".czi", "") + "threshold";
		saveAs("png", path + savetitle);
		close();
	}
waitForUser("Wow, that was a lot of worms, nice job!");