//Macro to split CZI into folders with split channel images within

//Written by Ed - used chaptGPT and FIJI to workout command syntax.


//Get the path from user and turn it into a list - get the length of the lsit for a loop
path = getDir("Please select the folder full of 'czi' files");
list_of_files = getFileList(path);
num_files = list_of_files.length;

//Make path to save DY96 image in
File.makeDirectory(path + "/DY96/");
dy96_path = path + "DY96\\";

//Taking a look at the debug window at the end of the loop
//print(error_maker);

//Make sure we don't cover the screen with images!
setBatchMode("True");


//Loop starts at 0, lasts while n is smaller than the length of files, and increases by one each loop
for(n=0; n<num_files; n++) {
	print(list_of_files[n]);
	//Only select files ending with CZI
	if (endsWith(list_of_files[n], ".czi")) {
		//Open file
		newpath = path + list_of_files[n];
		run("Bio-Formats Importer", "open=" + newpath + " autoscale color_mode=Default view=Hyperstack stack_order=XYCZT");
		title = getTitle();
		basename = replace(title, ".czi", "");
		//Channel 1 - DAPI
		Stack.setChannel(1);
		saveAs("png", path + basename + "DAPI.png");
		//Channel 2 - DY96
		Stack.setChannel(2);
		saveAs("png",dy96_path + basename + "DY96.png");
    	run("Close");
	}
}