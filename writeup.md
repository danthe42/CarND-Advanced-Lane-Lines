## Project writeup

## **Advanced Lane Finding Project**

Directories in of this repository:

- src/: python sources, the location of the source code
- camera_cal/: input chessboard images used for camera calibration
- cal_image_output/: camera calibration output images
- pickle/: serialized parameters used by the lane finding algorithm
- examples/: Udacity examples to help with the project
- test_images/: sample images to be used by the lane finding algorithm
- output_images/: output directory for images and project video

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[imagec1]: ./camera_cal/calibration2.jpg "Input"
[imagec2]: ./cal_image_output/calibration2.jpg "Detected corners"
[imagec3]: ./cal_image_output/undistorted_calibration2.jpg "Correctly undistorted image"
[image2]: ./test_images/test1.jpg "Before correction"
[image2post]: ./output_images/undistorted_test1.jpg "After correction"
[image3]: ./output_images/combined_test1.jpg "Binary Example"
[image4unw]: ./output_images/unwarped_straight_lines1.jpg "Undistorted with src"
[image4w]: ./output_images/warped_straight_lines1.jpg "Warped with dst"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/output_frame.jpg "Output"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. This writeup document is a part of my github project  [here](http://github.com/danthe42/CarND-Advanced-Lane-Lines/blob/master/writeup.md). It's describing the project, the sources and the algorithm used, and includes a few images to detail them.

First I want to detail the source files:

src/cvutils.py:  Library module. This file contains a few constants and general purpose image processing methods. It's used by (almost) every source files in the repository.

src/calibration.py:  Executable. Camera calibration codes are here

src/combination.py:  Executable. A PyQT GUI application with which it's easy to  interactively play with the thresholding/kernel size parameters when combining the binary filtered images.

src/lane.py: Library module. A class representing a given lane. Here we will use a "left" and a "right" lane. Algorithms for the low-level lane detections are part of this class.

src/car.py: Library module. The class representing the car, and containing member variables which are used during processing a frame. It contains also the 2 Lane objects, the combination parameters, the camera distortion matrixes and all internal images (warped, distorted, ...).

src/process_frame.py: Executable. Continuous stream (video) processing with the whole processing pipeline implementation. 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the separate src/calibration.py python file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The chessboards on the calibration images are 10x7, so I wanted to detect 9*6 internal corner points. As not all these necessary corners are visible in the calibration1, 4 and 5 images, those three inputs were ignored.   

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to all calibration images using the `cv2.undistort()` function. All the images with the recognized corners and the undistorted chessboards are in the 'cal_image_output' directory. The calibration output are the intrinsic matrix and the distortion coefficients. These are a property of the camera and it's lens. These values are written to the 'pickle/cal_result.p' file using the pickle python library.

As an example, here is one of the input with outputs: 

![alt text][imagec1]

![alt text][imagec2]

![alt text][imagec3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The first processing step of each images/image frames is always the correction of the distortion caused by the camera. This is achived by using the distortion coefficients calculated during calibration. To prove that the calibration was correct, the undistorted test images were calculated at the end of the previous calibration step (in src/calibration.py) and written to the 'output_images/undistorted*.jpg' files.

To demonstrate this step, here's a test image before, and after distortion correction:
![alt text][image2]

![alt text][image2post]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The best thresholding steps were determined after investigating many different combination of HLS channels, filters with different Sobel filters, and with different Gaussian blurs. 

I made a QT application with GUI, just for this purpose: 'src/combination.py'. It's possible to examine different combinations of parameters easily with that. It is using the ParameterTree QT widget from the pyqtgraph.parametertree module, which is a very convenient and useful widget for this purpose.

The combination of filter and threshold values I found to be the best for me were written to the 'pickle\combination_parameters.p' pickle file. I'm using the Absolute Sobel detection in X and Y direction, the gradient magnitude and the S, and L channels which I get after converting the image to the HLS colormap. 

Here's the output I got after processing the image just above.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

There is one more important processing step in the pipeline: Perspective transformation is necessary to transform the image so that it's seen from right above (birds eye view).  This code is in my utility module: 'src/cvutils.py' . The method name is warper(). The real calculation inside the function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  As the source and destination points mainly depends on the camera position relative to the  car, I chose to hardcode these in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

This is the reference undistorted image with the Source trapezoid drawn inside.

![alt text][image4unw]

And the unwarped image with the Dst rectangle:

![alt text][image4w]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I have 2 ways to detect the lanes: 

- The first method is that 
  - First, I create a histogram of the bottom half of the binary warped image, based on the x coordinate, so I can tell the number of activated points in a column. (in the bottom half of the image) 
  - The lanes should be somewhere around the maximum values in the histogram, the left lane on the left side of the screen, and the right lane on the right side. 
  - Then I try to detect the lanes using a sliding window method where I split the screen to 'n' horizontal bands, and going from bottom to top, I'm placing fixed sized windows for the lanes. 
  - The center 'x' of the window in the bottom band is determined by the histogram method previously described.
  - The window center on the next window will be the the mean 'x' coordinate of the activated pixels in the previous window if there was enough activated lane pixels in the previous window (this number is a hardcoded value). Otherwise the position of the window will be the same as the position of the window in the previous band. 
  - All activated pixels will be collected in an array, and at the end of this loop, a 2nd order polygon will be fitted on it with openCV's polyfit. This algorithm  will find the coefficients of the 2nd order polynomial which defines a curve where the sum of the squares of the distances of the points from the curve is the lowest. (Least squares polynomial fit)
  - This algorithm will find the lanes correctly most of the time, but it's 
    - Slow, because it's looking for the lanes in the whole image. 
    - Doesn't use the results of the lane findings in the past (the previous frames). 
  - This first method is in the file "car.py", the method's name is 'find_lanes".
- Because of these shortcomings, I use an other method to find the lanes when I have previously detected lanes:
  - The second method tries to "focus" the search of activated pixels around the previously detected lane curves with a predefined margin value. 
  - If the search is unsuccessful, we fall back and use the previous valid lane coefficients (same curve) at most for a given number of frames. (Maximum in the next half second) If that time expires, we fall back to the first method of the lane finding. 
  - This second method is in the file "car.py", the method's name is 'search_around_poly".

Az an example, here is the output of the more complicated, first method where the sliding windows are green rectangles, and the blue and left pixels are considered part of the two lanes.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the car position's distance from the lane center in "car.py" , method "calc_car_position" method. It will just be a very simple calculation after I have detected the lanes and if I consider the lane width to be 3.7m. (which is the standard lane width of highway lanes in the US)

The radius calculation is in file "lane.py" method "setcurrentfit" which method is used when the lane curve coefficients are changed. The mathematical background of the calculation of this value from the 2nd order polynomial coefficients of the lane curve is detailed here: https://www.intmath.com/applications-differentiation/8-radius-curvature.php

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in "process_frame.py" in the "process_frame" method. 

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4). It's awesome, just look at it ! 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think I've created the framework of a good, advanced lane detection method, from here it can be enhanced in many direction. A few ideas: 

- The combination.py can be used to create better thresholded binary images. It takes a lot of time and effort to experiment with it, but the resulting combination rule and the thresholds can enhance the lane detection quality a lot.
- Interestingly, from the 8 different types of binary images, few could be actually used. New types of filtered binary images could be invented. It's possible that the existing gradient/colormap channel/magnitude filters are not the most useful. In fact I had problems with these binary images during too bright light conditions, vary dark shadows with sharp border, patches of blobs in the binary image and so on. 
- Additional test images and videos can also help to tune these threshold/combination parameters. Especially is they are taken on different road surfaces, during different weather conditions, ...
- The difference between the last few lane polynomial coefficients ( diff member of object Lane ) could be used to detect sudden, wrong polygon fitting if the value is too big. This is a possible future improvement.
- Different lane width values can be handled. Based on the Country where we are, based on the type of the road the car is on, and using the given country's local rules for lane widths. These can be automatically detected by the car's built-in navigation application on the head unit.
- To reproduce failures of the detection in a continuous video stream is not too easy. It would be more convenient to use a GUI application where the developers could just pause the processing, go back to any frame and replay the processing again, with the ability to investigate all internal variables in the pipeline and the processing.
- The pipeline is not prepared to handle cases where there are very steep curves. There are cases when one lane is not even on the camera image for a long time. How should we handle this case ? 
- Normal, continuous lane markings can be handled. But how should we detect the interrupting and perpendicularly continuing lane curves in intersections ?  

Based on this list it can be concluded that there are many possible steps to be taken in the future, if me, or somebody wants to pursue this project further.

  