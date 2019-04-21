# **Traffic Sign Recognition**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## **Project Goal**
The goals of this project are the following:
* Load the traffic dataset
* Explore, summarize and visualize the data set
* Design, train and test a CNN model architecture
* Using the model to make predictions on new images downloaded from web
* Analyze the prediction(softmax) probabilities of the new images
* Future improvements - point of view

# **1. Pipeline - Traffic Images**
Pandas library is used to calculate summary statistics of the traffic signs dataset. Trainig features are a 4D array(number of  examples, width of an image, height of an image, color channels) of traffic sign images.

* The size of training set is: **34799,width:32, height:32, channels:3** 
* The size of validation set is:**4410**
* The size of test set is:**12630**
* The shape of a traffic sign image is:**Width of 32,a Height:32 and color channels:3(RBG)**
* The number of unique classes/labels in the data set is: **43 classes**

# **Observations**
1. The train data set is orderly packed.
2. The mapping between Class ID and Traffic Sign name are present in saperate file called "signname.csv".
3. Images are smaller in size and mostly front view.
4. Class distribution is not even - with few outliers like Class ID 1(Speed limit (20km/h),2(Speed limit (50km/h) and 13(Yield) followed by 12(Priority Road) abd 38(Keep right).

### **Eyeballing Images**
![image2](./examples/TrafficSample.png)

### **Class Distribution**
![image1](./examples/ClassDistribution.png)

# **Design and Test Model Architecture**
### **Pre-processing the dataset(normalization, grayscale and shuffling)**
The image dataset was normalized so that the data has mean zero and equal variance to help us to reduce the error. For image data, I used the below code to approximately normalize the dataset.

    	X_train=(X_train - 128)/ 128 
	
Image was converted to grayscale using this code

	X_train=np.sum(X_train/3, axis=3, keepdims=True)


### **Model Architecture**

My final model consisted of the following layers:

| Layer				|Description							| 
|:-----------------------------:|:-------------------------------------------------------------:| 
| Layer1: Convolution		| Input=32x32x1:Output=28x28x6:Filter:5x5:Stride:1x1:VALID	|
| Layer1:RELU			| Activation							|
| Layer1:Max Pooling		| Input=28x28x6:Output=14x14x6:ksize:2x2:Stride:2x2		|
| Layer2:Convolution 		| Input=14x14x6:Output=10x10x16:Filter:5x5:Stride:1x1:VALID	|
| Layer2:MaxPooling		| Input=10x10x16:Output=5x5x16:ksize:2x2:Stride:2x2		|
| Layer2:Flatten		| Input=5x5x16:Output=400					|
| Layer2:Dropout		| Keep Probability:0.5						|
| Layer3:FullyConnected		| Input=400:Output=120						|
| Layer3:RELU			| Activation							|
| Layer3:Dropout		| Keep Probability:0.5						|
| Layer4:FullyConnected		| Input=120:Output=84						|
| Layer4:RELU			| Activation							|
| Layer4:Dropout		| Keep Probability:0.5						|
| Layer5:FullyConnected		| Input=84:Output=43						|

### **Model Training**
Model training was iterative as the validation accuracy was below 80% inspite of chosing a well known architecture created by Pierre Sermanet and Yann LeCun (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I changed hyper parameters to make sure that model is generalized for testing. All adjustments on model architecture was due to overfitting i.e. a high accuracy on the training set but low accuracy on the validation set. Adition of dropout with slower learning rate helped to generalze the model. The final hyper parameters are:

	_epochs =150
	_batchSize=128
	_dropout= 0.5 #forcing few weights to play dump
	_learningRate = 0.0006 #slower rate 0.001 or 0.002 was resulting in less than 80%
#### **Final Results**
	Validation Accuracy = 0.974
	Test Accuracy with Model 2.0 = 0.952
	Test Accuracy with Model 1.0 = 0.944

#### **Model Tracking**
I implemented a tracking nomenclature since it was extremely difficult to keep track of model architecture, hyperpram and date & time. This approach helped me quickly organize my model training effort and save time. Please see below:






