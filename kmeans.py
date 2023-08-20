import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def kmeans_segment(folder_path, output_folder):
    image_files = os.listdir(folder_path)
    image_masked_list = []

    for img in tqdm(image_files, desc="Segmenting images", unit="img"):
        image_path = folder_path + img
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        height, width = image.shape
        image_2d = image.reshape((height * width, 1))

        n_clusters = 2  
        threshold = 110

        kmeans = KMeans(n_clusters=n_clusters , init=np.array([[0], [255]]))
        kmeans.fit(image_2d)

        cluster_centers = kmeans.cluster_centers_
        cluster_centers[cluster_centers <= threshold] = 10
        cluster_centers[cluster_centers > threshold] = 70
        kmeans.cluster_centers_ = cluster_centers
        
        cluster_labels = kmeans.predict(image_2d)
    

        # Reshape the cluster labels back to the original image shape
        segmented_image = cluster_labels.reshape((height, width))

        # Create a mask for each cluster
        segment_masks = []
        for cluster_id in range(n_clusters):
            mask = np.zeros_like(segmented_image, dtype=np.uint8)
            mask[segmented_image == cluster_id] = 255
            segment_masks.append(mask)

        image_masked1 = cv2.bitwise_and(image, image, mask=segment_masks[0])
        image_masked2 = cv2.bitwise_and(image, image, mask=segment_masks[1])

        blackPxNum_mask1 = np.count_nonzero([image_masked1<=70]) #number of black pixels
        blackPxNum_mask2 = np.count_nonzero([image_masked2<=70]) #number of black pixels

        if blackPxNum_mask1 < blackPxNum_mask2:
            chosen_mask = image_masked1
        else:
            chosen_mask = image_masked2
    
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_masked = cv2.bitwise_and(original, original, mask=chosen_mask)
        image_masked_list.append(image_masked)
        cv2.imwrite(output_folder + img, image_masked)
        
        pass 
    return image_masked_list
    