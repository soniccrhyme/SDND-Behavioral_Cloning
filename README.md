## **README**
---

## **Behavioral Cloning using a Driving Simulator and Keras**

### **Victor Roy**

[GitHub Link](https://github.com/soniccrhyme/SDND-Project_3)

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---

### Files, Prerequisites, Usage

#### 1. Files

Repository contains the following files:
* model.py containing the script which creates and trains a model
* drive.py for driving the car in autonomous mode
* model.h5 containing a pre-trained convolution neural network (CNN)

#### 2. Prerequisites

Python packages: see Udacity's package requirement list [here][https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/environment.yml]
The model.h5 file is compatible with Keras 2.0.4
The simulator for your respective OS can be found:
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
* [macOS][https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip]
* [Windows][https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip]

#### 3. Execution Instructions
Using the Udacity provided simulator as well as the drive.py and model.h5 provided here, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
and selecting Autonomous Mode. The current model was designed to work with Track #1 (on the left - the one through the desert).

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 CNN layers and 4 fully connected NNs. The first 3 CNN layers have increasing filter depths of 24, 32, & 48, using a kernel size of 5x5 and a stride of 2x2. The last two CNNs have filter depths of 64, using a kernel size of 3x3 and a stride of 1x1. All CNN layers include a ReLU activation to allow for nonlinearities.

The four fully-connected NNs have decreasing filter sizes of 100, 50, 10 and, finally, 1 - the last of which outputs the logits yields the steering angle. The first two fully-connected layers feature dropout layers with keep probabilities equal to 0.75.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

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
