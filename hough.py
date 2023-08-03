import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage import io
import os
from PIL import Image 
import PIL


def hough_transform(folder_path):
    image_files = os.listdir(folder_path)
    image_files = image_files[0:100] #getting only first 100 images in the folder 

    image_output = []
    skipped_images = []
    for image in image_files: 
        image_path = folder_path + image
        img= cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #increase brightness and contrast to make box more visible. 
        #im using the box seen in the xray image not the humerus itself
        contrast_factor = 4  # Increase the contrast by 50%
        brightness_factor = 200  # Do not change the brightness
        adjusted_image = cv2.convertScaleAbs(gray, alpha=contrast_factor, beta=brightness_factor)
        plt.imshow(adjusted_image, cmap='gray')

        edges = cv2.Canny(adjusted_image, 50, 150, apertureSize=3)
        lines_list =[]
        lines = cv2.HoughLinesP(
                    edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=100, # Min number of votes for valid line
                    minLineLength=5, # Min allowed length of line
                    maxLineGap=5 # Max allowed gap between line for joining them
                    )

        if lines is None:
            print("Skipping image, no lines found.")
            skipped_images.append(image_path)
            continue

        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2)])
            print("successful")
    

        # print(lines)
        angles = [line[0][1] for line in lines]
        # print(angles)

        average_angle = np.degrees(np.mean(angles))
        # if average_angle > 12000:
        #     average_angle = average_angle - (average_angle*0.50) 

        # print(average_angle)
        img_original = cv2.imread(image_path)
        height, width = img_original.shape[:2]
        center = (height // 2, width // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, average_angle, 1) #1 is image zoom
        rotated_image = cv2.warpAffine(gray, rotation_matrix, (width,height))
        image_output.append(rotated_image)
    return image_output, skipped_images