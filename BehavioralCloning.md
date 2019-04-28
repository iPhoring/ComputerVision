# **Behavioral Cloning** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## **Behavioral Cloning Project**
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## **Simulator Data Collection**
I collected the data (images) by driving the car in the Udacity simulator in training mode.The simulator environment provides two options: Training mode and Autonomous mode. 

1. Training Mode: The simulator will load where we can drive the car similar to video game.
2. Autonomous Mode: The autonomous mode is used to run the CNN trained model.

![image1](./examples/Simulator.png)

The input dataset comprised of 55032 images which is created by combining the four different datasets:  
1. Manually created dataset by driving the car in the center of the track in the clockwise direction 
2. Manually created dataset by driving the car close to the left edge of the track in the counterclockwise direction 
3. Manually created dataset by driving the car close to the right edge of the track in the clockwise direction  
4. Dataset provided by Udacity

In order to make the model unbiased, the car was driven towards the edge of right and left the sides of the lane in  clockwise and counterclockwise respectively in addition to driving at the center of the lane.

## **Data**
- The size of the training set is: 44025(80%),  height:160, width:320, channels:3 
- The size of the validation set is:11007(20%), height:160, width:320, channels:3 
- The number of classes/labels in the data set is: 1 class (Steering Angle)
### **Observation**
While investigating this large dataset, I noticed some factors which I considered, can cause confusion and affect the training of the model. 

![image2](./examples/SteeringAngle.png)

It is observed that the car has a tendency to drive straight which means there is a very little deviation in the steering angles. 
Besides, as there are shadows and darker images due to various reasons on the track, training the model without taking these into account, may produce erroneous model. 
As we know, a good model requires the car to steer correctly irrespective of the sides and turns and to identify the objects to drive it safely. To address these, I applied Augmentation techniques on the dataset. The steps are discussed in the following paragraphs.

### **Augmentation**
* Camera Angle correction:
To start with the Augmentation process, I first considered the correction of left and right steering angles.To do this I followed an empirical approach and set the left and right camera angle correction value as 0.15 and 0.2 respectively.

* Horizontal Flip:
A effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement.

      return np.fliplr(_image)
      _originalSteeringAngle = -1. *_originalSteeringAngle
      
* Darker Image & Shadow: I noticed that the track 1 has diffrent brightness and shadows in different areas while collecting the data. In order to train model for such conditions a random value was applied to RGB channel to make the image darker and a randon shadow was added to the image.

      #brightness
          _darkFactor = np.random.uniform(_lowValue, _highValue)    
          _image[:,:,2] = _image[:,:,2]*_darkFactor
    
      #shadow
         cols, rows = (_image.shape[0], _image.shape[1])    
        _topLeft = np.random.uniform(_lowValue, _highValue) * _image.shape[1]
        _bottomLeft = np.random.uniform(_lowValue, _highValue) * _image.shape[1]    
        if np.random.random_sample() <= 0.6:
            _bottomRight = _bottomLeft - np.random.uniform() *(rows-_bottomLeft)
            _topRight = _topLeft - np.random.uniform() *(rows-_topLeft)
        else:        
            _bottomRight = _bottomLeft + np.random.uniform() *(rows-_bottomLeft)
            _topRight = _topLeft + np.random.uniform() *(rows-_topLeft) 
        _poly = np.asarray([[[_topLeft,0],[_bottomLeft, cols],[_bottomRight, cols],[_topRight,0]]], dtype=np.int32)       
        ...
        return cv2.addWeighted(_image.astype(np.int32), _alphaWeight, _srcImageCopy, _weightFactor, 0).astype(np.uint8)
    
* Blurred: We used the smallest size and the fastest graphical quality for capturing data. In order to train model for such conditions a gaussian filter was used to blurr the image.

      def _blurred(_image,_alpha=0.6):
          return ndimage.gaussian_filter(_image, _alpha)
* Translation: In order to remove streering angle bias 0.0(driving stright) I used openCV function to translate image pixcels and adjusted streeing angle to sync with the translation. After few permutaions a adjustment of 0.002 was used as the steering angle adjustment per pixcel.
      _newSteeringAngle = _originalSteeringAngle +( _transX * _steeringAngleShift)
      _transMatrix = np.float32([[1, 0, _transX],[0, 1, _transY]])
      _image = cv2.warpAffine(_image, _transMatrix, (_width,_height))


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
