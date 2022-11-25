# -*- coding: utf-8 -*-
# Advanced zoom example. Like in Google Maps.
# It zooms only a tile, but not the whole image. So the zoomed tile occupies
# constant memory and not crams it with a huge resized image for the large zooms.
import random
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import asksaveasfile, askopenfilename
from PIL import Image, ImageTk

# from detectron2.utils.visualizer import Visualizer, ColorMode
# from Buchnearer import random_color
import cv2
import numpy as np
import pickle
import copy

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
    ]
).astype(np.float32).reshape(-1, 3)

def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret

class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
       Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')

class Zoom_Advanced(ttk.Frame):
    ''' Advanced zoom of the image '''
    def __init__(self, mainframe): #, im_file, result_file
        ''' Initialize the main Frame '''
        im_file = askopenfilename(title = "Select image file")
        result_file = askopenfilename(title = "Select result file")
        self.image = cv2.imread(im_file)
        with open(result_file, 'rb') as dbfile:      
            self.result = pickle.load(dbfile) 
        self.root = mainframe
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Zoom with mouse wheel')
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')
        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.__scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.__scroll_x)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.master.title('Buchearer Corrector')
        self.master.geometry('1000x1000')
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__draw_oval_press)
        self.canvas.bind('<B1-Motion>', self.__draw_oval_press_on_move)
        self.canvas.bind('<ButtonRelease-1>', self.__release_draw_oval)
        self.canvas.bind('<ButtonPress-3>', self.__delete)
        self.canvas.bind('<ButtonPress-2>', self.__move_from)
        self.canvas.bind('<B2-Motion>',     self.__move_to)
        self.canvas.bind('<MouseWheel>', self.__wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # only with Linux, wheel scroll up
        self.canvas.bind('<Key>', self.__keystroke)
        
        self.masks = self.result['masks']
        self.boxes = self.result['boxes']
        self.scores = self.result['scores']
        # self.image1 = Image.open(path1)  # open image
        # self.image2 = Image.open(path2)  # open image
        # self.visualizer = Visualizer(self.image[:, :, ::-1],
        #                   scale=1.0, 
        #                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        #     )
        self.assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(len(self.masks))]
        # self.visimage = self.visualizer.overlay_instances(masks=self.masks, assigned_colors=self.assigned_colors)
        # self.segimage = Image.fromarray(self.visimage.get_image()[:, :, ::-1])
        self.segimage = self.__add_all_masks(self.image, self.masks, self.assigned_colors)
        self.imno = 1 # flag used for switch orignal or segmented images
        self.delete_mode = -1 # flag used for switch between select mode and delete mode.
        self.height, self.width, _ = self.image.shape
        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude
        self.__previous_state = 0  # previous state of the keyboard
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        # Plot some optional random rectangles for the test purposes
        minsize, maxsize, number = 5, 20, 10
        self.show_image()
        self.canvas.focus_set()  # set focus on the canvas
        

    def __scroll_y(self, *args, **kwargs):
        ''' Scroll canvas vertically and redraw the image '''
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def __scroll_x(self, *args, **kwargs):
        ''' Scroll canvas horizontally and redraw the image '''
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def __delete(self, event):
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        y = (event.x+bbox2[0]-bbox1[0])/self.imscale
        x = (event.y+bbox2[1]-bbox1[1])/self.imscale
        for i in range(len(self.masks)):
            if [int(x),int(y)] in self.masks[i].tolist():
                self.segimage = self.__delete_mask(self.segimage, self.masks[i], self.assigned_colors[i])
                del self.masks[i]
                del self.assigned_colors[i]
                del self.scores[i]
                del self.boxes[i]
                self.show_image()
                return

    def __draw_oval_press(self, event):
        if self.delete_mode == -1:
        # save mouse drag start position
            self.oval_start_x_ = self.canvas.canvasx(event.x)
            self.oval_start_y_ = self.canvas.canvasy(event.y)
            self.oval_start_x = copy.deepcopy(event.x)
            self.oval_start_y = copy.deepcopy(event.y)
            # create rectangle if not yet exist
            self.oval = self.canvas.create_oval(self.oval_start_x_, self.oval_start_y_, self.oval_start_x_+1, self.oval_start_y_+1, outline='red', width=2)
        else:
            self.__delete(event)

    def __draw_oval_press_on_move(self, event):
        if self.delete_mode == -1:
            self.oval_end_x_ = self.canvas.canvasx(event.x)
            self.oval_end_y_ = self.canvas.canvasy(event.y)
            self.oval_end_x = copy.deepcopy(event.x)
            self.oval_end_y = copy.deepcopy(event.y)
            # expand rectangle as you drag the mouse
            self.canvas.coords(self.oval, self.oval_start_x_, self.oval_start_y_, self.oval_end_x_, self.oval_end_y_)

    def __release_draw_oval(self, event):
        if self.delete_mode == -1:
            self.canvas.delete(self.oval)
            bbox1 = self.canvas.bbox(self.container)  # get image area
            bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
            bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                     self.canvas.canvasy(0),
                     self.canvas.canvasx(self.canvas.winfo_width()),
                     self.canvas.canvasy(self.canvas.winfo_height()))
            start_y = (self.oval_start_x+bbox2[0]-bbox1[0])/self.imscale
            start_x = (self.oval_start_y+bbox2[1]-bbox1[1])/self.imscale
            end_y = (self.oval_end_x+bbox2[0]-bbox1[0])/self.imscale
            end_x = (self.oval_end_y+bbox2[1]-bbox1[1])/self.imscale
            self.segimage = self.__add_new_mask(self.segimage, start_x, start_y, end_x, end_y)
            self.show_image()

    def __move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def __wheel(self, event):
        ''' Zoom with mouse wheel '''
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale        /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale        *= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        print(event.keysym)
       
        self.__previous_state = event.state  # remember the last keystroke state
        if event.keysym == 'm': # switch between mask view and original view
            self.imno = -self.imno
            self.show_image()
        if event.keysym == 'r': # refresh the mask image
            self.assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(len(self.masks))]
            self.segimage = self.__add_all_masks(self.image, self.masks, self.assigned_colors)
            self.show_image()
        if event.keysym == 's': # save mask file
            data = [('Result Files', '*.result')]
            file = asksaveasfile(filetypes = data, defaultextension = '.result')
            result = {}
            result['masks'] = self.masks
            result['boxes'] = self.boxes
            result['scores'] = self.scores
            with open(file.name, 'wb') as outfile:
                pickle.dump(result, outfile)
        if event.keysym == 'q': # open a new image
            self.master.destroy()
            self.__init__(tk.Tk())

        if event.keysym == 'Escape': # open a new image
            self.master.destroy()
            quit()

        if event.keysym == 'd': # switch between select mode and delete mode
            if self.delete_mode == -1:
                self.text = tk.Label(self.root, text='Delete Mode', font=("Arial",12))
                self.text.grid(row=0, column=0, sticky='NW')
            else:
                self.text.destroy()
                
            self.delete_mode = -self.delete_mode
            self.show_image()

    def show_image(self, event=None):
        ''' Show image on the Canvas '''
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)   # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            if self.imno == 1:
                image = Image.fromarray(np.clip(self.segimage,0,255).astype('uint8')).crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            else:
                image = Image.fromarray(self.image).crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __add_all_masks(self, image, masks, color):
        mask_image = copy.deepcopy(image)
        for i in range(len(masks)):
            # x,y = masks[i].nonzero()
            # for (a,b) in zip(x,y):
            for (a,b) in masks[i]:
                mask_image[a,b,:] = mask_image[a,b,:]*0.5 + color[i]*255*0.5
        return mask_image

    def __delete_mask(self, image, mask, color):
        mask_image = image
        # x,y = mask.nonzero()
        # for (a,b) in zip(x,y):
        for (a,b) in mask:
            mask_image[a,b,:] = 2*(mask_image[a,b,:] - color*255*0.5)
        return np.clip(mask_image,0,255)

    def __add_new_mask(self, image, start_x, start_y, end_x, end_y):
        mask_image = image
        # mask = np.zeros(self.masks[0].shape)
        mask = []
        c_x = (start_x+end_x)/2
        c_y = (start_y+end_y)/2
        a2 = np.square((end_x-start_x)/2)
        b2 = np.square((end_y-start_y)/2)
        for x in range(int(start_x), int(end_x+1)):
            for y in range(int(start_y), int(end_y+1)):
                if np.square(x-c_x)/a2+np.square(y-c_y)/b2 < 1:
                    # mask[x,y] = True
                    mask.append([x,y])
        self.masks.append(np.array(mask))
        self.assigned_colors.append(random_color(rgb=True, maximum=1))
        self.boxes.append(np.flip(np.concatenate([np.max(mask, axis=0),np.min(mask, axis=0)])))
        self.scores.append(1)
        # x,y = self.masks[-1].nonzero()
        # for (a,b) in zip(x,y):
        for (a,b) in mask:
            mask_image[a,b,:] = mask_image[a,b,:]*0.5 + (self.assigned_colors[-1]*255)*0.5
        return mask_image



# path1 = '/home/xupan/Projects/Buchnearer/test images/test_image_by_age/D9/20210115_LSR1_D9_flat.lif - 20210115_LSR1_flat_1_DAPI.png'
# path2 = '/home/xupan/Projects/Buchnearer/test images/test_image_by_age/D9/Prediction/20210115_LSR1_D9_flat.lif - 20210115_LSR1_flat_1_DAPI.result'
# path2 = '/home/xupan/Projects/Buchnearer/test2.result'
root = tk.Tk()
app = Zoom_Advanced(root) #, path1, path2
root.mainloop()