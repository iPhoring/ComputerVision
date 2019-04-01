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


# **Camera Calibration**
The code for this step is contained in the first few cell of the IPython notebook located in [AdvancedLaneFinding](./CarNDAdvancedLaneFindingV6.ipynb).

The very 1st step is to prepare "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. The basic assumption is that the chessboard is fixed on the (x, y) plane at z=0 and all object points are same every image. Imagge points will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

Finally, use the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:
# **Pipeline (Test Images)**
## **Gradients and color transforms**
A combination of color and gradient thresholds is used to generate a binary image.

![image1](./test_images/CameraCali.png)

## **Perspective Transformation**
Perspective transform is done using cv2.warpPerspective transform function. The function takes as inputs an image (img), as well as source (src_points) and destination (dst_points) points along with the camera tranformation matrix. I chose the hardcode the source and destination points in the following manner:

src_points=np.float32([[585,455],[702,455],[1200,720],[160,720]]) #by trial and error method

offset = 200 # offset for dst points

#Grab the image shape
img_size = (gray.shape[1], gray.shape[0])

dst_points = np.float32([[offset, 0],
                     [img_size[0]-offset, 0],
                     [img_size[0]-offset, img_size[1]],
                     [offset, img_size[1]]])
## **Color and Thresholding**
Gradient and color transformation are achived by taken sobel x i.e derivative in x and then thresholding the image to convert it into a binary image.

sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1


### **Color channel**
I converted the image to HLS color space in addition to RBG to detect lane under diffrent light condition like shadows, missing markings,miss alignment and if the road top is not blacktop.

![image2](./test_images/GradientsColor)

### **Region of Interest**
With trial and error method four points where identified to mark the lanes. 

Additionally, a horizontal sliding window approach is used to find lane in case of sharp turns. 
