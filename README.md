# HoughTransform_xray

### I tried using Hough Transform to automatically rotate humerus xray images, but i encoutered the following issues: <br>
1. it is only successfull if there is a visible full rectangle in the image. <br>
2. The angle of the image is determined from the average of angles from all lines detected in the image. This messes up the computation for the rotation
<br>
<p align="center">
    <img src="https://github.com/ronocara/HoughTransform_xray/blob/d526c7eae83ccdd0088eef9868990b841c90d2c1/output/output_codeV1.png" width="300">
   <br> Issue with the first version of the code
</p>

<br>

### To have better results i have implemented the following steps in order: <br>

   1. apply thresholding to focus on the important part with the humerus<br>
   2. using cv2.findcontours , draw a rectangle on the focused object<br>
   3. using Hough Lines, get only the longest line detected and get it's angle. <br>
   4. Get rotation matrix using the angle of the longest line and the center of the image <br>
   5. using cv2.wrapAffine rotate the original image based on the computed rotation matrix<br>
   6. Center image<br>

   <br>
<p align="center">
    <img src="https://github.com/ronocara/HoughTransform_xray/blob/d526c7eae83ccdd0088eef9868990b841c90d2c1/output/output.png">
   <br> Rotated vs Original image (latest version)
</p>

   <br>
<p align="center">
    <img src="https://github.com/ronocara/HoughTransform_xray/blob/d526c7eae83ccdd0088eef9868990b841c90d2c1/output/output2.png">
   <br> image rotated and centered
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