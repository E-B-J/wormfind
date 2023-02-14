# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:20:24 2023

@author: ebjam
"""

import tkinter
import tkinter as tk
from tkinter import messagebox
import pickle
import os
import numpy as np
#import cv2

# Import the package if saved in a different .py file else paste 

start_folder = "C:/Users/ebjam/Downloads/gui testers-20230213T211340Z-001/gui testers/"
full_image_pickle = "full_image_results.pickle"
index_segmentation_record = os.path.join(start_folder, full_image_pickle)
file = open(index_segmentation_record,'rb')
seg_record = pickle.load(file)

segs = seg_record["results"][2].masks.segments

def transpose_segmentation(bbox, segmentation):
    minx = bbox[0]
    miny = bbox[1]
    for i in segmentation:
        '''
        i[0] * w turns decimal location into real location in input image. 
        i.e. coordinate of x = 0.567 on an image 1000 pixels wide would be pixel 567
        
        '- minx' and '-miny' transposes segmentation to the bbox rather than the full image
        '''
        i[0] = i[0] - minx
        i[1] = i[1] - miny
    #Segmentation is now transposed to bbox
    return(segmentation)

#need to get the segs and image title from pickle.

class ScrollableImage(tkinter.Frame):
    def __init__(self, master=None, **kw):
        self.images = kw.pop('images', [])
        self.current_image_index = 0
        self.image = self.images[self.current_image_index]
        self.image_titles = kw.pop("im_titles", [])
        self.segs = kw.pop('segmentations', [])
        self.seg = self.segs[self.current_image_index] #Masks for speicific image
        self.masks = self.seg.masks.segments
        self.polygon_visibility = True
        self.right_click_coords = []
        super(ScrollableImage, self).__init__(master=master, **kw)
        self.cnvs = tkinter.Canvas(self, highlightthickness=0, **kw)
        self.cnvs.create_image(0, 0, anchor='nw', image=self.image)
        if self.polygon_visibility:
            for seg in self.masks:
                points = []
                for point in seg:
                    point=point.tolist()
                    point[0] = point[0] * 2752
                    points.append(point[0])
                    point[1] = point[1] * 2208
                    points.append(point[1])
                self.cnvs.create_polygon(points, fill='red', tag="mask")
        # Vertical and Horizontal scrollbars
        self.v_scroll = tkinter.Scrollbar(self, orient='vertical', width=20)
        self.h_scroll = tkinter.Scrollbar(self, orient='horizontal', width=20)
        # Grid and configure weight.
        self.cnvs.grid(row=0, column=0,  sticky='nsew')
        self.h_scroll.grid(row=1, column=0, sticky='ew')
        self.v_scroll.grid(row=0, column=1, sticky='ns')
        self.rowconfigure(0, weight=4)
        self.columnconfigure(0, weight=4)
        # Set the scrollbars to the canvas
        self.cnvs.config(xscrollcommand=self.h_scroll.set, 
                           yscrollcommand=self.v_scroll.set)
        # Set canvas view to the scrollbars
        self.v_scroll.config(command=self.cnvs.yview)
        self.h_scroll.config(command=self.cnvs.xview)
        # Assign the region to be scrolled 
        self.cnvs.config(scrollregion=self.cnvs.bbox('all'))
        self.bind_all("<Right>", self.change_image_f)
        self.bind_all("<Left>", self.change_image_r)
        self.bind_all("<m>", self.change_mode)
        self.bind_all("<Button-1>", self.select_polygon)
        self.bind_all("<Delete>", self.delete_polygon)
        self.bind_all("<s>", self.save_worms)
        self.bind_all('<Button-3>', self.draw_polygon)


    def change_image_f(self, event):
        self.current_image_index = (self.current_image_index + 1)
        self.image = self.images[self.current_image_index]
        self.cnvs.delete("all")
        self.cnvs.create_image(0, 0, anchor='nw', image=self.image)
        self.seg = self.segs[self.current_image_index] #Masks for speicific image
        self.masks = self.seg.masks.segments
        for seg in self.masks:
            points = []
            for point in seg:
                point=point.tolist()
                point[0] = point[0] * 2752
                points.append(point[0])
                point[1] = point[1] * 2208
                points.append(point[1])
                
            self.cnvs.create_polygon(points, fill='red', tags="mask")
        
        self.cnvs.config(scrollregion=self.cnvs.bbox('all'))
        
    def change_image_r(self, event):
        if self.current_image_index > 0:
            self.current_image_index = (self.current_image_index - 1)
            self.image = self.images[self.current_image_index]
            self.cnvs.delete("all")
            self.cnvs.create_image(0, 0, anchor='nw', image=self.image)
            self.seg = self.segs[self.current_image_index] #Masks for speicific image
            self.masks = self.seg.masks.segments
            for seg in self.masks:
                points = []
                for point in seg:
                    point=point.tolist()
                    point[0] = point[0] * 2752
                    points.append(point[0])
                    point[1] = point[1] * 2208
                    points.append(point[1])
                    
                self.cnvs.create_polygon(points, fill='red', tags="mask")
            
            self.cnvs.config(scrollregion=self.cnvs.bbox('all'))
        
    def change_mode(self, mevent):
        if self.polygon_visibility:
            self.cnvs.delete("mask")
            self.cnvs.delete("points")
            self.polygon_visibility = False
        else:
            self.seg = self.segs[self.current_image_index] #Masks for speicific image
            self.masks = self.seg.masks.segments
            for seg in self.masks:
                points = []
                for point in seg:
                    point=point.tolist()
                    point[0] = point[0] * 2752
                    points.append(point[0])
                    point[1] = point[1] * 2208
                    points.append(point[1])
                self.cnvs.create_polygon(points, fill='red', tags = "mask")
            self.polygon_visibility = True
    
    def draw_polygon(self, event):
        if not self.right_click_coords:
            self.right_click_coords = []
        x, y = ((event.x, event.y))
        self.right_click_coords.append((x, y)) # !!! Format for polygon draw!
        #Show the point clicked on!
        self.cnvs.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", tags = "line")
        #Draw line between points if possible
        if len(self.right_click_coords) >= 2:
            self.cnvs.create_line(self.right_click_coords[-2][0], self.right_click_coords[-2][1], x, y, fill="red", tags="line")  
        #Deal with ending polygon
        startpoint = self.right_click_coords[0]
        if len(self.right_click_coords) >= 3:
            if (x - startpoint[0])**2 + (y - startpoint[1])**2 < 100:
                # Close polygon and remove temporary line
                self.cnvs.create_line(self.right_click_coords[-1][0], self.right_click_coords[-1][1],
                                  startpoint[0], startpoint[1], fill="red", tags="line")
                self.right_click_coords.pop()
                
                self.cnvs.create_polygon(self.right_click_coords, outline="red", tags="mask")
                self.cnvs.delete("line")
                self.right_click_coords = []
            
    def delete_polygon(self, event):
            # Get the selected polygon
            self.cnvs.delete("selected")

            
    def select_polygon(self, event):
        # Get the item that was clicked
        item = self.cnvs.find_withtag("current")
        # Check if the item is a polygon
        if "mask" in self.cnvs.gettags(item):
            # Deselect any previously selected polygons
            self.cnvs.dtag("selected", "all")
            # Add the "selected" tag to the clicked polygon
            self.cnvs.addtag_withtag("selected", item)
            # Change the fill color of the selected polygon to yellow
            self.cnvs.itemconfig(item, fill="yellow")
            coords = self.cnvs.coords(item)
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            # create a new set of canvas objects for the points
            selected_points = []
            for x, y in points:
                point = event.widget.create_oval(x-2, y-2, x+2, y+2, fill="red", tags = ["selected", "points"])
                selected_points.append(point)

        else:
            # Deselect any previously selected polygons
            self.cnvs.dtag("selected", "all")
            self.cnvs.itemconfig("mask", fill="red")

    def save_worms(self, event):
        annotations = {"single_worms": []}
        title = self.image_titles[self.current_image_index]
        #DY96img = cv2.imread(start_folder + "DY96/" + title)
        polygons = self.cnvs.find_withtag("mask")
        save_gon = []
        worm_no = 0
        # Loop through the polygons and add their coordinates to the data dictionary
        for polygon in polygons:
            annotation = {}
            save_title = title[:-4] +"worm_" + str(worm_no)
            coords = self.cnvs.coords(polygon)
            points = np.array([[coords[i], coords[i+1]] for i in range(0, len(coords), 2)])
            save_gon.append(points)
            annotation["title"] = title
            annotation["wormID"] = save_title
            annotation["bbox"] = self.cnvs.bbox(polygon)
            annotation["segmentation"] = points
            annotation["transposed_segmentation"] = transpose_segmentation(annotation["bbox"], annotation["segmentation"])
            annotations["single_worms"].append(annotation)
            #worm_crop = DY96img[int(annotation["bbox"][1]):int(annotation["bbox"][3]), int(annotation["bbox"][0]):int(annotation["bbox"][2])]
            #cv2.imwrite(start_folder + "DY96/"+ save_title+ ".png", worm_crop)
            with open(start_folder + "DY96/" + title[:-4] + ".result", "wb") as f:
                pickle.dump(annotation, f, protocol=pickle.HIGHEST_PROTOCOL)
            worm_no +=1
        messagebox.showinfo("Save", "file saved")            

root = tk.Tk()

images = [f for f in os.listdir(start_folder) if f.endswith(".png") ]

img_load = [tk.PhotoImage(file = start_folder +"/" + f) for f in images]
results = seg_record["results"]

image_window = ScrollableImage(root, images=img_load, im_titles = images, segmentations = results, width=2752, height=2208)
image_window.pack()

root.mainloop()