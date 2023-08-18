# HoughTransform_xray

### Following steps in order: <br>

   1. apply thresholding to focus on the important part with the xray <br>
   2. draw a rectangle on the focused object. Used cv2.findcontours and find the best fitting rectangle on the object     
      detected <br>
       - the rectangle is binary, where the rectangle is white. This will be used as the mask for segmentation and rotation <br>
   3. using Hough Lines, get only the longest line detected in the rectangle and get it's angle. <br>
       - if no line was detected the image is saved in a seperate array
       - if the image is already upright no rotation is done
   4. Get rotation matrix using the angle of the longest line and the center of the image <br>
   5. using cv2.wrapAffine rotate the original image based on the computed rotation matrix<br>
       - made sure height is always longer so image is always vertical
       - houghline rotation output is horizontal so implememted 90 degree rotation

<br>
<p align="center">
    <img src="https://github.com/ronocara/HoughTransform_xray/blob/4e38665eed28c6c466f9b51061e16438953b3f10/output/output4.png">
   <br> segmentation, rotation, and centering process
</p>

   <br>
<p align="center">
    <img src="https://github.com/ronocara/HoughTransform_xray/blob/4e38665eed28c6c466f9b51061e16438953b3f10/output/output.png">
   <br> results on different images
</p>


## <b>Issues encoutered</b><br>
   <br>
<p align="center">
    <img src="https://github.com/ronocara/HoughTransform_xray/blob/d526c7eae83ccdd0088eef9868990b841c90d2c1/output/output3.png" width="400">
   <br> Still having issues with centering images. It does not work on all the images
</p>

   <br>
<p align="center">
    <img src="https://github.com/ronocara/HoughTransform_xray/blob/d526c7eae83ccdd0088eef9868990b841c90d2c1/output/outpu-flawed.png" width="400">
   <br> There are also instances where the humerus inside the rectangle is not upright. Only the rectangle has the correct upright position and not the humerus
</p>
