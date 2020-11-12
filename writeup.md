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
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. This writeup document is a part of my github project  [here](http://github.com/danthe42/CarND-Advanced-Lane-Lines/blob/master/writeup.md). It's describing the project, the sources and the algorithm used, and includes a few images to detail them.

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

The combination of filter and threshold values I found to be the best for me were written to the 'pickle\combination_parameters.p' pickle file. Here's the output I got after processing the image just above.  

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

![alt text][image4unw]

![alt text][image4w]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
