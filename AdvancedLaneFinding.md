# **Advanced Lane Finding**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


# *Camera Calibration* 
The code for this step is contained in the first few cell of the IPython notebook located in "./examples/example.ipynb".

The very 1st step is to prepare "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. The basic assumption is that the chessboard is fixed on the (x, y) plane at z=0 and all object points are same every image. Imagge points will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

Finally, use the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:
