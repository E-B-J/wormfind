# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:57:13 2024

@author: ebjam
"""

import cv2, os
import matplotlib.pyplot as plt
import numpy as np

def getWorms(image):
    img = cv2.imread(img_path + image)
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit = 80, tileGridSize = (32, 32))
    converted_img = clahe.apply(grayimg)
    blurred_image = cv2.GaussianBlur(converted_img, (5, 5), 2)  # Kernel size of (5, 5) with sigma=2
    min_value = 100
    max_value = 255

    # Clip the values to set the background and foreground
    clipped_image = np.clip(blurred_image, min_value, max_value)
    threshold_value = 175
    _, thresholded_image = cv2.threshold(clipped_image, threshold_value, 255, cv2.THRESH_BINARY)
    kernel_open = np.ones((3, 3), np.uint8)  # You can adjust the kernel size as needed
    mask_opened = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel_open, iterations=5)
    mask_eroded = cv2.erode(mask_opened, kernel_open, iterations=2)

    # Sharpen the image
    sharpening_kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
    sharpened_image = cv2.filter2D(mask_eroded, -1, sharpening_kernel)
    sharp_save_handle = binary_path + image[:-3] + "png"
    cv2.imwrite(sharp_save_handle, sharpened_image)
    plt.imshow(sharpened_image)
    plt.show()
    return
img_path = "E:/toronto_microscopy/ixmc/July28LiquidlotN2vsAWR73/July28LiquidlotN2vsAWR73/TimePoint_1/dapi/one_field/"
binary_path = img_path + "binary/"
os.makedirs(binary_path, exist_ok=True)
images = [q for q in os.listdir(img_path) if q.endswith("TIF")]
for image in images:

    getWorms(image)