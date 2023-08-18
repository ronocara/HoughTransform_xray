import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import PIL
from skimage.filters import threshold_otsu, sobel
from tqdm import tqdm


def houghT_rotate(folder_path, output_folder, outliers_path, th_less):
    image_files = os.listdir(folder_path)
    outliers = os.listdir(outliers_path)

    for i, x in enumerate(outliers):
        if ".png.png" in x:
            x = x.replace(".png.png",".png")
            outliers[i] = x
    
    for y in outliers:
        if y in image_files:   
            image_files.remove(y)

    non_centered = []
    image_output = []
    # skipped_images = []

    for image in tqdm(image_files, desc="Processing images", unit="image"): 
        image_path = folder_path + image
        im_gray = np.array(Image.open(image_path).convert('L'))

        #get rectangle mask
        mask, image_masked, no_mask = get_rect(im_gray, th_less) # th_less = percentage to lessen threshold

        #We use HoughLines now to rotate the rectangle
        edges = cv2.Canny(mask, 50, 150, apertureSize=3) #get contours of the rectangle
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
            # print("Skipped images:", len(skipped_images))
            # skipped_images.append(image_path)
            # skipped_image = cv2.imread(image_path)
            # cv2.imwrite(skipped_image_path+image, skipped_image)
            # continue
            centered_image = image_masked
            
        else:
            for points in lines:
                # Extracted points nested in the list
                x1,y1,x2,y2=points[0]
                length = np.sqrt((x2 - x1)^2 + (y2 - y1)^2)
                # # Draw the lines joing the points
                # # On the original image
                # cv2.line(mask,(x1,y1),(x2,y2),(0,255,0),2)
                # # Maintain a simple lookup list for points
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
            # img_original = cv2.imread(image_path)
            height, width = im_gray.shape[:2]
            center = (width // 2, height // 2)
            # if (angle_degrees <=0.6 and angle_degrees >=0):

            rotated_image = np.uint8(image_masked)
            rotated_mask = mask #the white rectangle
    

            # if (angle_degrees <=0.6 and angle_degrees >=0):
            #     rotated_image = np.uint8(rotated_image)
            #     rotated_mask = mask
            #     if height < width:
            #         rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #         rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            #if the rect is already upright, will not rotate
            #angles withint 75 - 90 considered upright. having issues with rotation with angles 75-89
            if (angle_degrees >= -90 and angle_degrees <= -75) or (angle_degrees <= 90 and angle_degrees >= 75 ) or (angle_degrees <=0 and angle_degrees>= 0.5) : 
                if (angle_degrees <= 90 and angle_degrees >= 75 ):
                    rotated_image = cv2.rotate(np.uint8(image_masked), cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated_mask =cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif (angle_degrees >= -90 and angle_degrees <= -75):
                    rotated_image = cv2.rotate(np.uint8(image_masked), cv2.ROTATE_90_CLOCKWISE)
                    rotated_mask =cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            
            #if rectangle not upright
            else: 
                rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1) # 1 is the image zoom
                #making sure the height is always longer so rectangle is always vertical 
                if height < width:
                    rotated_image = cv2.warpAffine(image_masked, rotation_matrix, (width, height))
                    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height))
                else:
                    rotated_image = cv2.warpAffine(image_masked, rotation_matrix, (height, width))
                    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (height, width))

            #rotation after warpAffine always horizontal. will make it vertical with elbow on bottom
            if angle_degrees < -0.6:
                rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle_degrees > 0.6:
                rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
                rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_90_CLOCKWISE)

            #to make sure the image is always vertical'
            mask_height, mask_width = rotated_mask.shape[:2]
            if mask_height < mask_width:
                if angle_degrees < -0.6:
                    rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif angle_degrees > 0.6:
                    rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
                    rotated_mask = cv2.rotate(rotated_mask, cv2.ROTATE_90_CLOCKWISE)
            
        # if there's no mask/rectangle detected will not auto center image
        #mask is the basis for the centering
        if no_mask == True:
            centered_image = rotated_image
        else:
            centered_image = center_object(rotated_image , rotated_mask)

        
        image_output.append(centered_image)
        non_centered.append(rotated_image) #for double checking

        #check if still needs to be inverted 
        centered_image = np.array(centered_image)
        blackPxNum = np.count_nonzero([centered_image<=100]) #number of black pixels
        whitePxNum = centered_image.size - blackPxNum 

        if blackPxNum < whitePxNum:
            centered_image = cv2.bitwise_not(centered_image)

        cv2.imwrite(output_folder+image, centered_image)
        
    return image_output , non_centered

def get_rect(im_gray, th_less):
    #get image threshold
    threshold = threshold_otsu(im_gray)
    threshold -= threshold * th_less #lessen threshold   
    bina_image = im_gray < threshold
    inverted_bina_image = np.logical_not(bina_image)

    # invert image
    inverted_binary_image = Image.fromarray(np.uint8(inverted_bina_image) * 255)
    
    #removed image background after thresholding (not perfect)
    background_removed_image = np.zeros_like(im_gray)
    background_removed_image[inverted_bina_image]  = im_gray[inverted_bina_image]
    background_removed_image = Image.fromarray(background_removed_image)

    #get contours of binary image
    contours, _ = cv2.findContours(np.uint8(background_removed_image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    original_image = np.array(background_removed_image)
    mask = np.zeros_like(original_image)
    no_mask = False
    cv2.drawContours(mask, [np.int0(best_rect)], 0, (255, 255, 255), cv2.FILLED) # get the mask rectangle
    image_masked = cv2.bitwise_and(original_image, original_image, mask=mask) #getting only object insde the rectangle

    #sometimes it cannot detect a rectangle. so we use the original image
    if all(element == 255 for row in mask for element in row):
        mask = image_masked
        # to know if there was no mask detected or not
        #if no mask, will not center image
        no_mask = True 

    return mask, image_masked, no_mask

def center_object(rotated_image, rotated_mask):
    w, h, x, y=0,0,0,0
    hh, ww = rotated_mask.shape

    # get the contours of the rotated image
    contours = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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