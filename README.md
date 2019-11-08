# Udacity-Self-Driving-Car
Train a Self-Driving Car in the Udacity's Self Driving Simulator

## Dependencies
- Matplotlib (https://pypi.org/project/matplotlib/)
- Numpy (https://pypi.org/project/numpy/)
- Flask (https://pypi.org/project/Flask/)
- Socketio (https://pypi.org/project/socketio/)
- Tensorflow (https://pypi.org/project/tensorflow/)
- Keras (https://pypi.org/project/Keras/)
- Scikit-Learn (https://pypi.org/project/scikit-learn/)


## Neural Network Model

<p align="center">
<img src="https://github.com/crypto-code/Udacity-Self-Driving-Car/blob/master/assets/model.JPG" align="left" />   </p>


## Data Augmentation

To increase the number of samples with different steering angles, the dataset is augmented by flipping images and using cameras from the side of the car with a modified course-correcting angles. These augmentations also help the neural network learn to recover from disruptions.

<p align="center">
<img src="https://github.com/crypto-code/Udacity-Self-Driving-Car/blob/master/assets/augment.JPG" align="left" />   </p>

## Pretrained Model
Downlad the pretrained model [here](https://drive.google.com/open?id=1cbL44GLz6JfH04NR03Jt95W8X3fPH4Bj)
