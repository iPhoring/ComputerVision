import os
import csv
import cv2
from math import ceil
from sklearn.utils import shuffle
import numpy as np
import sklearn
from scipy import ndimage        
from sklearn.model_selection import train_test_split
#------------------------helper functions--------------------------------------------------
def _collectDataSamples(_paths=[['./mydata/',0],['./data/',1]],_cameraAngle=['center'],_cameraAngleCorrectionL=0.25,_cameraAngleCorrectionR=0.25):
    samples = []
    for _path in _paths:
        #print('Looping thru the files')
        with open(_path[0]+'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            if _path[1]==1: #input with header row
                next(reader, None)
            for line in reader:                                
                if 'center' in set(_cameraAngle):
                    samples.append([[_path[0]+'IMG/'+line[0].split('/')[-1],float(line[3])],[0]])
                if 'left'in set(_cameraAngle):
                    samples.append([[_path[0]+'IMG/'+line[1].split('/')[-1],float(line[3])+_cameraAngleCorrectionL],[0]])
                if 'right'in set(_cameraAngle):
                    samples.append([[_path[0]+'IMG/'+line[2].split('/')[-1],float(line[3])-_cameraAngleCorrectionR],[0]])
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return  train_samples,validation_samples
#-----------------------------collecting and combining data--------------------------------------------------------------------------
_header=1
train_samples,validation_samples =_collectDataSamples(_paths=[['./mydata/',0], #Counter clockwise center
                                                              ['./data/',1], #Udacity
                                                              ['./dataturn/',0],#Recovery clockwise left
                                                              ['./dataturn1/',0] #Recovery counter clockwise right
                                                              ],
                                                      _cameraAngle=['center','left','right'],
                                                      _cameraAngleCorrectionL=0.15,_cameraAngleCorrectionR=0.2)
#--------------------------Image Augmentation------------------------------------------------------------------------------------------
def _dark(_image,_lowValue=0.2, _highValue=0.75):
    _image=cv2.cvtColor(_image, cv2.COLOR_RGB2HSV)
    _darkFactor = np.random.uniform(_lowValue, _highValue)    
    _image[:,:,2] = _image[:,:,2]*_darkFactor
    return cv2.cvtColor(_image, cv2.COLOR_HSV2RGB)

def _blurred(_image,_alpha=0.6):
    return ndimage.gaussian_filter(_image, _alpha)

def _horizontalFlip(_image):
    return np.fliplr(_image)

def _randomShadow(_image, _lowValue=0.6, _highValue=0.75):    
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
    _weightFactor = np.random.uniform(_lowValue, _highValue)
    _alphaWeight = 1 - _weightFactor
    _srcImageCopy = np.copy(_image).astype(np.int32)
    cv2.fillPoly(_srcImageCopy, _poly, (0, 0, 0))
    return cv2.addWeighted(_image.astype(np.int32), _alphaWeight, _srcImageCopy, _weightFactor, 0).astype(np.uint8)

def _translateImage(_image, _originalSteeringAngle, _rangeX=100, _rangeY=10, _steeringAngleShift=0.002):
    _height, _width,ch = _image.shape
    _transX = _rangeX * (np.random.rand() - 0.5)
    _transY = _rangeY * (np.random.rand() - 0.5)
    
    _newSteeringAngle = _originalSteeringAngle +( _transX * _steeringAngleShift)
    _transMatrix = np.float32([[1, 0, _transX],[0, 1, _transY]])
    _image = cv2.warpAffine(_image, _transMatrix, (_width,_height))
    return _image, _newSteeringAngle

def _augmentImage(_image,_originalSteeringAngle,_steeringAngleShift=0.002,_probability=0.6):    
    if np.random.random_sample() <= _probability: 
        _image = _horizontalFlip(_image)
        _originalSteeringAngle = -1. *_originalSteeringAngle
    if np.random.random_sample() <= _probability:
        _image=_dark(_image)
    if np.random.random_sample() <= _probability: 
        _image=_randomShadow(_image,_lowValue=0.6, _highValue=0.75)
    if np.random.random_sample() <= _probability: 
        _image=_blurred(_image,_alpha=0.8)
    if np.random.random_sample() <= _probability:
        _image, _originalSteeringAngle = _translateImage(_image, _originalSteeringAngle,_steeringAngleShift)
    return _image, _originalSteeringAngle
#-----------------------------------------------------end-------------------------------
def generator(samples, batch_size=32,_augmentation=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                _imageFullFileName = batch_sample[0][0]
                _augmentation=batch_sample[1][0]                
                _image=ndimage.imread(_imageFullFileName)                
                #============pre-processing============
                #print('right after reading',center_image.shape)
                #center_image=center_image[60:-20, :, :] #removing the top and car front rows
                #print('right after crop',center_image.shape)             
                #_image=cv2.cvtColor(_image, cv2.COLOR_RGB2YUV) #converting to YUV color space as needed by NVIDIA
                #print('finally',center_image.shape)
                #=============
                _angle = float(batch_sample[0][1])
                if _augmentation==1:
                    _image, _augmentedSteeringAngle=_augmentImage(_image,_angle,_steeringAngleShift=0.002,_probability=0.6)
                    images.append(_image)
                    angles.append(_augmentedSteeringAngle)
                elif  _augmentation==0:                    
                    images.append(_image)
                    angles.append(_angle)                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
#-----------------------------Modelling---------------------------------
# Set our batch size
batch_size=50
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size,_augmentation=True)
validation_generator = generator(validation_samples, batch_size=batch_size,_augmentation=True)


from keras.models import Sequential, Model
from keras.layers import Lambda,Cropping2D,Input
from keras.layers import Dense, Dropout, Flatten, Convolution2D
from keras.layers import Conv2D, MaxPooling2D
from keras.backend import tf as ktf
from keras.callbacks import ModelCheckpoint, EarlyStopping

#implementing LeNet with dropout - Val Acc:67%
def _LeNet_dropout():
    _signature='_LeNet_dropout'
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
    #layers
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(160,320,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu')) 
    model.add(Dense(1))
    return model,_signature

def _NVIDIA_modified():
    #https://arxiv.org/pdf/1704.07911.pdf - used this for reference
    #https://arxiv.org/pdf/1604.07316v1.pdf - used this for reference
    _signature='_NVIDIA_Modified'
    #custom pre-processing of images
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(lambda _image: ktf.image.resize_images(_image, (66, 200))))
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(80,320,3)))

    #model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2,2))) #strided convolutions 2×2 stride and a 5×5 kernel/filter
    model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2,2)))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))#non-strided convolution with a 3×3 kernel size in the last two convolutional layers
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu')) #non-strided convolution with a 3×3 kernel size in the last two convolutional layers
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))    
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model,_signature

#_model,_signature=_LeNet_dropout() #moving to NVIDIA as LeNet is too slow to process
_model,_signature=_NVIDIA_modified()

#model checkpoint
_checkpoint = ModelCheckpoint(filepath='model'+_signature+'checkpoint.h5', monitor='val_loss', save_best_only=True)
_stopper = EarlyStopping(monitor='val_acc', min_delta=0.0003, patience=5)

_modelDesign={
    'Paths': [   #data file paths
                "['./mydata/',0], #Udacity",
                "['./data/',1], #counter clockwise center",
                "['./dataturnC/',0],#recovery clockwise",
                "['./dataturnR/',0] #recovery counter clockwise'", 
             ],
    'CameraAngles':['center','left','right'],
    'SteeringCorrection':['0.2L','0.25R'],
    'BatchSize':'50',
    'SteeringAngleShift':'0.002',
    'AugmentationProbability':'0.6',
    'AugmentationFlag':'True',
    'ModelSignature':_signature,
    'ModelArchitecture':
        [
        "Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3))",
        'Lambda(lambda x: x/127.5-1.0, input_shape=(80,320,3))',
        "Conv2D(filters=24, kernel_size=(5, 5), activation='elu', strides=(2,2))",
        "Conv2D(filters=36, kernel_size=(5, 5), activation='elu', strides=(2,2))",
        "Conv2D(filters=48, kernel_size=(5, 5), activation='elu', strides=(2,2))",
        "Conv2D(filters=64, kernel_size=(3, 3), activation='elu')",
        "Conv2D(filters=64, kernel_size=(3, 3), activation='elu')",
        "Flatten()",
        "Dense(100,activation='elu')) ",
        "Dense(50, activation='elu'))",
        "Dense(10, activation='elu'))",
        "Dense(1))"
        ]
}
_model.compile(loss='mse', optimizer='adam',metrics=['accuracy']) #accuracy is not the target. Focus is on loss.Predicting steering angle(single continous value)- regression


_historyObject=_model.fit_generator(train_generator,
    steps_per_epoch=ceil(len(train_samples)/batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples)/batch_size),
    epochs=5, verbose=1,callbacks=[_checkpoint,_stopper])

print(_modelDesign)
print(_historyObject.history['loss'],_historyObject.history['val_loss'],_historyObject.history['acc'],_historyObject.history['val_acc'])
_model.save('modelNVIDIA_V4.h5')