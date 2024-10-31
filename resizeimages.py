# resize images to fit within 800x600 while keeping the aspect ratio same
import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

random.seed(1211)

image_folders = sorted(glob.glob('/home/preyum.kumar/CollegeWork/computer vision/Assignment 3/Images/*'))
for folder in image_folders:
    images = sorted(glob.glob(folder + '/*'))
    for image in images:
        img = cv2.imread(image)
        
        # Get original dimensions
        h, w = img.shape[:2]
        
        # Desired dimensions
        # target_w, target_h = 150, 150
        target_w, target_h = 200, 200
        
        # Calculate the aspect ratio
        aspect_ratio = w / h
        
        # Determine new dimensions while keeping the aspect ratio
        if aspect_ratio > 1:  # Width is greater than height
            new_w = target_w
            new_h = int(target_w / aspect_ratio)
        else:  # Height is greater than or equal to width
            new_h = target_h
            new_w = int(target_h * aspect_ratio)
        
        # Resize the image
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save the resized image
        cv2.imwrite(image, img_resized)
        print(f'{image} resized to {new_w}x{new_h}')