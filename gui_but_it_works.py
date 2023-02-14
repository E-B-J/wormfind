# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:20:24 2023

@author: ebjam
"""

import tkinter
import tkinter as tk
import pickle
import os
# Import the package if saved in a different .py file else paste 

start_folder = "C:/Users/ebjam/Downloads/gui testers-20230213T211340Z-001/gui testers/"
full_image_pickle = "full_image_results.pickle"
index_segmentation_record = os.path.join(start_folder, full_image_pickle)
file = open(index_segmentation_record,'rb')
seg_record = pickle.load(file)

segs = seg_record["results"][2].masks.segments



#need to get the segs and image title from pickle.

class ScrollableImage(tkinter.Frame):
    def __init__(self, master=None, **kw):
        self.images = kw.pop('images', [])
        self.current_image_index = 0
        self.image = self.images[self.current_image_index]
        self.segs = kw.pop('segmentations', [])
        self.seg = self.segs[self.current_image_index] #Masks for speicific image
        self.masks = self.seg.masks.segments
        super(ScrollableImage, self).__init__(master=master, **kw)
        self.cnvs = tkinter.Canvas(self, highlightthickness=0, **kw)
        self.cnvs.create_image(0, 0, anchor='nw', image=self.image)
        
        for seg in self.masks:
            points = []
            for point in seg:
                point=point.tolist()
                point[0] = point[0] * 2752
                points.append(point[0])
                point[1] = point[1] * 2208
                points.append(point[1])
            self.cnvs.create_polygon(points, fill='red')

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
        #self.cnvs.bind_class(self.cnvs, "<MouseWheel>", self.mouse_scroll)
        
        self.bind_all("<Right>", self.change_image)

    def change_image(self, event):
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
                
            self.cnvs.create_polygon(points, fill='red')
        self.cnvs.config(scrollregion=self.cnvs.bbox('all'))

 
root = tk.Tk()

# PhotoImage from tkinter only supports:- PGM, PPM, GIF, PNG format.
# To use more formats use PIL ImageTk.PhotoImage
images = [f for f in os.listdir(start_folder) if f.endswith(".png") ]

img_load = [tk.PhotoImage(file = start_folder +"/" + f) for f in images]
results = seg_record["results"]

#img = tk.PhotoImage(file="C:/Users/ebjam/Downloads/gui testers-20230213T211340Z-001/gui testers/72hrn2i_25u-08.png")

image_window = ScrollableImage(root, images=img_load, segmentations = results, width=2752, height=2208)
image_window.pack()

root.mainloop()