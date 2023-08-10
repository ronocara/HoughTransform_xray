import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import PIL
from skimage.filters import threshold_otsu, sobel

def houghT_rotate(folder_path, output_folder, skipped_image_path):
    image_files = os.listdir(folder_path)

    non_centered = []
    image_output = []
    skipped_images = []

    for image in image_files: 
        image_path = folder_path + image
        im_gray = np.array(Image.open(image_path).convert('L'))
        
        drawn_image, _ = get_rect(im_gray)

        #We use HoughLines now to rotate the rectangle
        edges = cv2.Canny(drawn_image, 50, 150, apertureSize=3) #get contours of the rectangle
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
            skipped_image = cv2.imread(image_path)
            cv2.imwrite(skipped_image_path+image, skipped_image)
            continue
            
        for points in lines:
            # Extracted points nested in the list
            x1,y1,x2,y2=points[0]
            length = np.sqrt((x2 - x1)^2 + (y2 - y1)^2)
            # Draw the lines joing the points
            # On the original image
            cv2.line(drawn_image,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            # lines_list[0].append([(x1,y1),(x2,y2)])
            lines_list.append(length) #get the length of the line 

        #get the longest line detected
        longest_line = max(lines_list)
        longest_line_index = lines_list.index(longest_line)
        longest_line = lines[longest_line_index][0]

        #get the angle of the longest line
        angle_radians = np.arctan2(longest_line[3] - longest_line[1], longest_line[2] - longest_line[0])
        angle_degrees = np.degrees(angle_radians)
        
        #rotate the original image 
        img_original = cv2.imread(image_path)
        height, width = img_original.shape[:2]
        center = (width // 2, height // 2)
        #if the image is already upright, will not rotate
        if angle_degrees == -90 or angle_degrees == 90 : 
            if angle_degrees == 90:
                rotated_image = cv2.rotate(im_gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle_degrees == -90:
                rotated_image = cv2.rotate(im_gray, cv2.ROTATE_90_CLOCKWISE)
        else: 
            rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1) # 1 is dimage zoom
            rotated_image = cv2.warpAffine(im_gray, rotation_matrix, (height,width))

        if angle_degrees <= 0:
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle_degrees > 0:
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
        
        
        centered_image = center_object(rotated_image)
        image_output.append(centered_image)
        non_centered.append(rotated_image)
        cv2.imwrite(output_folder+image, centered_image)

    return image_output , skipped_images , non_centered

def get_rect(im_gray):
    #get image threshold
    threshold = threshold_otsu(im_gray)
    threshold -= threshold*0.20
    bina_image = im_gray < threshold
    inverted_bina_image = np.logical_not(bina_image)

    # invert image
    inverted_binary_image_pil = Image.fromarray(np.uint8(inverted_bina_image) * 255)
    
    #get contours of binary image
    contours, _ = cv2.findContours(np.uint8(inverted_bina_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #find the best rectangle that will fit to the humerus
    best_rect = None
    best_rect_area = 0

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box_area = rect[1][0] * rect[1][1]

        if box_area > best_rect_area:
            best_rect = box
            best_rect_area = box_area

    #draw the rectangle on the original image. then make the inside white
    #now we have a white rectangle
    original_image = np.array(inverted_binary_image_pil)
    drawn_image = original_image.copy()
    cv2.drawContours(drawn_image, [np.int0(best_rect)], 0, (255, 255, 255), -1)

    return drawn_image, best_rect

def center_object(rotated_image):
    hh, ww = rotated_image.shape

    # get the contours of the rotated image
    contours = cv2.findContours(rotated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)

    # recenter
    startx = (ww - w)//2
    starty = (hh - h)//2
    result = np.zeros_like(rotated_image)
    result[starty:starty+h,startx:startx+w] = rotated_image[y:y+h,x:x+w]
        
    return result


#sources:
# image center: https://stackoverflow.com/questions/59525640/how-to-center-the-content-object-of-a-binary-image-in-python
# hough transform: https://medium.com/wearesinch/correcting-image-rotation-with-hough-transform-e902a22ad988