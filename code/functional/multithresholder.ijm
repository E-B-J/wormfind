path = getDirectory("Choose image dir");
list_of_files = getFileList(path);
num_files = list_of_files.length;
wormno = 1
for(n=0; n<num_files;n++)
	{
		file = list_of_files[n];
		open(path+file);
		run("8-bit");
		run("Threshold...");
		waitForUser("Press ok after applying threshold.");
		setOption("BlackBackground", true);
		run("Convert to Mask");
		savetitle = file.replace(".png", "") + "threshold";
		saveAs("png", path + savetitle);
		close();
		print("\\Clear");
		print("That was worm number: ");
		print(wormno);
		wormno = wormno + 1;
	}
waitForUser("Wow, that was a lot of worms, nice job!");