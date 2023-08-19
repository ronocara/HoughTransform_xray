# HoughTransform_xray

### Following steps done in order: <br>

1. remove the outliers listed in the anomalies folder<br>
2. get rectangle mask<br>
3. use HoughLines now to get lines of the rectangle<br>
	3.a get contours of the rectangle<br>
	3.b if no rectangle detected , keep it as is<br>
	3.c get the longest line detected<br>
	3.d get the angle of the longest line<br>
4. Rotation of the rectangle<br>
   the rectangle mask will be used for segmentation, anything outside the rectangle is considered background<br>
	4.a If there is no rectangle detected. will keep image as is
   4.b if the rectangle is horizontal, rotate 90 degrees<br>
	4.c if rectangle is already upright no need for rotation matrix<br>
	4.d if not upright or horizontal, get rotation matrix and compute for proper upright position<br>
	4.e check if height is longer than width, if not interchange the width with the height<br>
5. centering of image<br>
	5.a if there is no mask  /rectangle detected, will not center<br>
	5.b if there's a rectangle center image based on the rectangle position<br>

### To run do, the following:
1. make 3 folders :<br>
   - output_0, output_1, output_3
   - this is because there is iteration. the output_0 will be used as input for the second run and so on. 
2. Use houghT.ipynb to run <br>
   - folder_path = MURA humerus images
   - txt_path = to_invert.txt path (list images that needs to be inverted)
   - output_paths = paths of the output folders


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
