# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py contains model. Preprocessing of data is part of model.
* train.py is the script to train by model defined in model.py.
* drive.py for driving the car in autonomous mode and save images.
* model.h5 containing a trained weight of model defined in model.py. 
* video.py is for create video from images contained in a folder.
* video.mp4 shows model driving performance.
* video_simulator.mp4 is screen recording (4x speed) of simulator run on autonomous mode.
* writeup.md summarizing the results.

[//]: # (Image References)

[model]: ./examples/model_plot.png
[center1]: ./examples/center_1.jpg
[center2]: ./examples/center_2.jpg
[left_center1]: ./examples/left_center_1.jpg
[left_center2]: ./examples/left_center_2.jpg
[right_center1]: ./examples/right_center_1.jpg
[right_center2]: ./examples/right_center_2.jpg
[left_turn]: ./examples/left_turn.jpg
[right_turn]: ./examples/right_turn.jpg

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for model and preprocessing is part of model. 

The train.py shows the pipeline I used for training and validating the model, and it contains comments to explain how the 
code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

This modle is "MobileNet" model of "ImageNet" without last fully connected layers. Its first 40 layes are freezed 
(can not be trained). As "MobileNet" supports some specific images sizes so image preprocessing is added as input layes 
Lambda function. Finally flatten the "MobileNet" output and added two Dense layer.

Preprocessing has following two steps:
* Resize input image to support "MobileNet" supported input size. We have used (160, 320, 3) to (160, 160, 3).
* "MobileNet" model requires normalized image which pixel value should be between -1, 1. Used keras supported preprocess 
function of "MobileNet" model.


#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting after flatten and first dense layer. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The train.py uses an adam optimizer to train model in model.py. Tried diffrent learning rate but 0.001 worked best, 
which is also default of adam optimizer.


#### 4. Appropriate training data

I have used given training data. Only use images of central camera. 
As we have less number of right curved lane date, horizontally fliped images and its negative stearing mesurnments is used
to increase training data and get sufficient right turn images.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Choosen smallest sized imagenet model supported in keras-2.0.8. Removed the final layer instead of that 
added two dense layers. To deal with overfitting two dropout layers is also added. To feed our image input size 
lambda layer for preprocessing is added (details are in model.py). 

Chossen pretrained model "MobileNet" with "ImageNet" weight. Freezed first fourty layers. As intitial layers has generaral 
features (i.e. edges), which moldel has to learn in any trainings. In our cases we had to learn lane lines edges. Purpose of 
trnsfer learning was to reduce train time and have good edges features of "MobileNet" model.

Although number of epochs is 30, but model stop improving validation loss after 7 only. Validation loss in our case was 
0.097 (mean square error). Its traing was pretty fast. 

Finally model performance was tested on simulator. I have tested on maximum supported speed (approx 30 MPH) of simulator. 
Its runs preety well. Its means relatively big size of model has good enough response time to handle this speed.


#### 2. Final Model Architecture

Our final model architecture can be shown by following image.

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here are example images 
of center lane driving:

![alt text][center1] ![alt text][center2] 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the 
vehicle would learn to recover if deviated from center of lane. 

* Example images to recover from left

![alt text][left_center1] ![alt text][left_center2]

* Example images to recover from right

![alt text][right_center1] ![alt text][right_center2]

To augment the data set, flipped images and take negative of stearing angle mesurnment for fliped images. By this 
agumentation technique training example become twice and also balance examples for left and right turn lane lines.
An example of fliped image.

![alt text][right_turn] ![alt text][left_turn]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
Number of epochs was 30 with early stoping callbacks. Found that in 7 epochs validation error stoped improving. 
Uses adam optimizer with learning rate 0.001. Validation mse error was 0.097.




