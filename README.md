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
<img src="https://github.com/crypto-code/Udacity-Self-Driving-Car/blob/master/assets/model.jpg" align="middle" />   </p>

```
Training samples       13,692
Validation samples      2,282

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 65, 320, 64)       4864
_________________________________________________________________
activation_1 (Activation)    (None, 65, 320, 64)       0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 106, 64)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 106, 128)      204928
_________________________________________________________________
activation_2 (Activation)    (None, 21, 106, 128)      0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 35, 128)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 35, 256)        819456
_________________________________________________________________
activation_3 (Activation)    (None, 7, 35, 256)        0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 11, 256)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 5632)              0
_________________________________________________________________
dense_1 (Dense)              (None, 512)               2884096
_________________________________________________________________
activation_4 (Activation)    (None, 512)               0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               131328
_________________________________________________________________
activation_5 (Activation)    (None, 256)               0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 257
=================================================================
Total params: 4,044,929
Trainable params: 4,044,929
Non-trainable params: 0
_________________________________________________________________
```

## Data Augmentation


## Pretrained Model
Downlad the pretrained model [here](https://drive.google.com/open?id=1cbL44GLz6JfH04NR03Jt95W8X3fPH4Bj)
