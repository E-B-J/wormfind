# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:27:23 2023

@author: ebjam
"""

from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import pickle
import os

#Load info from last piece of code - dict opject, item 1 - list of files, item 2 = list of results from theose files
start_folder = "C:/Users/ebjam/Downloads/gui testers-20230213T211340Z-001/gui testers"
full_image_pickle = "full_image_results.pickle"
index_segmentation_record = os.path.join(start_folder, full_image_pickle)
file = open(index_segmentation_record,'rb')
seg_record = pickle.load(file)
imglist = seg_record["image_titles"]

#%%Make photoimage objects
# Create the window and canvas
root = Tk()

images = []
for img in imglist:
    image = ImageTk.PhotoImage(Image.open(os.path.join(start_folder, img)))
    images.append(image)
print(image.width())
print(image.height())
#Plotted masks storage
plotted_masks = []
# Initialize the current image index
current_image_index = 0
unsaved_changes = 0

# Load the first image
image_view = images[current_image_index]
canvas = Canvas(root, width=image_view.width(), height=image_view.height())
canvas.grid(row=0, column=0)
# Display the image on the canvas
canvas.create_image(0, 0, image=image_view, anchor=NW)

h=Scrollbar(root, orient='horizontal')
h.grid(row=1, column=0, sticky='ew')

root.mainloop()


#Get masks
masks = seg_record["results"][current_image_index].masks
segs = masks.segments

for seg in segs:
    for point in seg:
        point[0] = point[0] * 2752
        point[1] = point[1] * 2208
    xpoints = [point[0] for point in seg]
    ypoints = [point[1] for point in seg]
    canvas_item = canvas.create_polygon(xpoints, ypoints, fill='red')
    plotted_masks.append(canvas_item)
#%%

for mask in masks:
    segment = mask.segment
    for point in segment:
        point[0] = point[0] * images[0].width
        point[1] = point[1] * images[0].height
    xpoints = [point[0] for point in segment]
    ypoints = [point[1] for point in segment]
    canvas_item = canvas.create_polygon(xpoints, ypoints, fill='red')
    plotted_masks.append(canvas_item)

# Define a variable to store the currently selected polygon
selected_polygon = None
points = []
add_mode =0
scale = 1

# Define a function to handle mouse clicks on the canvas
def on_canvas_click(event):
    global selected_polygon
    global unsaved_changes
    global add_mode
    global points
    
    if add_mode == 0:
        # Find the polygon closest to the mouse click
        closest_polygon = None
        closest_distance = float('inf')
        for canvas_item in plotted_masks:
            distance = canvas.find_closest(event.x, event.y, halo=5, start=canvas_item)[0]
            if distance < closest_distance:
                closest_polygon = canvas_item
                closest_distance = distance
    
        # If a polygon was found, select it
        if closest_polygon is not None:
            selected_polygon = closest_polygon
    
            # Change the color of the selected polygon to green
            canvas.itemconfig(selected_polygon, fill="yellow")
            for point in selected_polygon:
                canvas.create_oval(point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5, fill="orange")
            
        if unsaved_changes == 0:
            unsaved_changes = 1
            save_button = Button(root, text = "Save worms", command = save_worms)
            save_button.grid(row=1, column = 2)
    elif add_mode == 1:
        x, y = ((event.x, event.y))
        points.append((x, y))
        #Show the point clicked on!
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red")
        #Draw line between points if possible
        if len(points) >= 2:
            canvas.create_line(points[-2][0], points[-2][1], x, y, fill="red")   
# Bind the mouse click event to the canvas
canvas.bind("<Button-1>", on_canvas_click)

def delete_selected_polygon():
    global selected_polygon
    global plotted_masks
    global unsaved_changes
    # If a polygon is selected, delete its canvas item
    if selected_polygon is not None:
        canvas.delete(selected_polygon)
        plotted_masks.remove(selected_polygon)
        selected_polygon = None
        unsaved_changes = 1

# Bind the delete key to the function to delete the selected polygon
root.bind("<Delete>", delete_selected_polygon)


#Zoom functions from chatGPT
def zoom_in(event):
    global scale
    scale *= 1.1
    canvas.scale("all", event.x, event.y, scale, scale)

def zoom_out(event):
    global scale
    scale *= 0.9
    canvas.scale("all", event.x, event.y, scale, scale)

canvas.bind("<Button-4>", zoom_in) # bind zoom_in function to mouse scroll up event
canvas.bind("<Button-5>", zoom_out) # bind zoom_out function to mouse scroll down event


#Set up functions to draw new polygons
"""
Need to turn on edit mode, record mouse click, plot polygon, and update plotted_masks

"""


#Turn on edit mode
def addnew():
    global points
    global add_mode
    if add_mode == 0:
        add_mode = 1
    elif add_mode == 1:
        add_mode = 0
        points.clear()
        

canvas.bind("<Key-n>", addnew)

#Function to add newly drawn polygon to list of polygons - False negative curation
def on_key(event):
    global unsaved_changes
    global points
    if event.keysym == "Return":
        polygon = points.copy()
        polygon.append(polygon[0]) #Close the loop
        polygon_id = canvas.create_polygon(polygon, fill="", outline="red")
        plotted_masks.append((polygon))
        points.clear()
        unsaved_changes = 1

# Bind the key event to the on_key function
canvas.bind("<Key>", on_key)




# Define the function to be called when the "Next" button is clicked
def next_image():
    global current_image_index
    global image
    global seg_record
    global unsaved_changes
    # Clear any existing polyggon shapes
    canvas.delete("all")
    plotted_masks = []

    # Increment the current image index
    current_image_index = (current_image_index + 1)

    # Load the next image
    image = Image.open(images[current_image_index])

    # Convert the image to a PhotoImage object
    image = ImageTk.PhotoImage(image)

    # Display the image on the canvas
    canvas.create_image(0, 0, image=image, anchor=NW)

    #Load masks for image
    masks = seg_record["results"][current_image_index].masks

    # Iterate through each polygon
    for mask in masks:
        segment = mask.segment
        for point in segment:
            point[0] = point[0] * images[0].width
            point[1] = point[1] * images[0].height
        xpoints = [point[0] for point in segment]
        ypoints = [point[1] for point in segment]
        canvas.create_polygon(xpoints, ypoints, fill='red')
    unsaved_changes = 0
    status = Label(root, text = "Image " + str(current_image_index+1) + " of " + str(len(imglist)+1), bd=1, relief=SUNKEN, anchor=E)
    status.grid(row=2, column = 0, columnspan = 4)
    
def prev_image():
    global current_image_index
    global image
    global seg_record
    global unsaved_changes
    # Clear any existing polyggon shapes
    canvas.delete("all")
    plotted_masks = []
    
    # Increment the current image index
    current_image_index = (current_image_index - 1)

    # Load the next image
    image = Image.open(images[current_image_index])

    # Convert the image to a PhotoImage object
    image = ImageTk.PhotoImage(image)

    # Display the image on the canvas
    canvas.create_image(0, 0, image=image, anchor=NW)

    #Load masks for image
    masks = seg_record["results"][current_image_index].masks

    # Iterate through each polygon
    for mask in masks:
        segment = mask.segment
        for point in segment:
            point[0] = point[0] * images[0].width
            point[1] = point[1] * images[0].height
        xpoints = [point[0] for point in segment]
        ypoints = [point[1] for point in segment]
        canvas_item = canvas.create_polygon(xpoints, ypoints, fill='red')
        # Add the canvas item to the list of canvas items
        plotted_masks.append(canvas_item)
    unsaved_changes = 0
    status = Label(root, text = "Image " + str(current_image_index+1) + " of " + str(len(imglist)+1), bd=1, relief=SUNKEN, anchor=E)
    status.grid(row=2, column = 0, columnspan = 4)
    
def save_worms():
    global plotted_masks
    global seg_record
    global images
    global current_image_index   
    global unsaved_changes
    if unsaved_changes == 1:
        dumpdict = {}
        #Crop worms here
        
        validated_segmentation = []
        for mask in plotted_masks:
            seg_points = []
            for point in mask:
                point[0] = point[0] / images[0].width
                point[1] = point[1] / images[0].height
                point = tuple(point)
                seg_points.append(point)
            validated_segmentation.append(seg_points)
        dumpdict['title'] = images[current_image_index]
        dumpdict['segmentations'] = validated_segmentation
        with open(title[-4] + "validated.result", w) as dumpfile:
            pickle.dump(dumpdict, dumpfile)
        unsaved_changes = 0
        save_button = Button(root, text = "Worms saved!", state = DISABLED, command = save_worms)
        save_button.grid(row=1, column = 2)
        
# Create the "Next", "Previous", exit, and save button
next_button = Button(root, text=">>", command=next_image)
exit_button = Button(root, text = "Exit GUI", command = root.quit)
save_button = Button(root, text = "Save worms", command = save_worms)
prev_buttton = Button(root, text = "<<", command = prev_image)
exit_button.grid(row = 1, column = 0)
next_button.grid(row=1, column=1)
save_button.grid(row=1, column = 2)
prev_button.grid(row=1, column=3)

#Add status tracker
status = Label(root, text = "Image " + str(current_image_index+1) + " of " + str(len(imglist)+1), bd=1, relief=SUNKEN, anchor=E)
status.grid(row=2, column = 0, columnspan = 4)



# Start the Tkinter event loop
root.mainloop()