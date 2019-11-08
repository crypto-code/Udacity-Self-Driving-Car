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
<img src="https://github.com/crypto-code/Udacity-Self-Driving-Car/blob/master/assets/model.JPG" />   </p>


## Data Augmentation

To increase the number of samples with different steering angles, the dataset is augmented by flipping images and using cameras from the side of the car with a modified course-correcting angles. These augmentations also help the neural network learn to recover from disruptions.

<p align="center">
<img src="https://github.com/crypto-code/Udacity-Self-Driving-Car/blob/master/assets/augment.JPG" align="middle" />   </p>

## Usage

To train your own model, first change line 57 in train.py to the directory of your training samples
```
# Change this line to your training data
    simulation_logs = ['data/t1_first/driving_log.csv', 'data/t1_backwards/driving_log.csv', 'data/t1_forward/driving_log.csv']
```
Then run,
```
python train.py
```
You can also downlad the pretrained model [here](https://github.com/crypto-code/Udacity-Self-Driving-Car/releases/tag/v1.0)

To drive in autonomous mode, you can use the below available arguments
```
usage: drive.py [-h] [--model MODEL] [--image_folder [IMAGE_FOLDER]]
                [--maxspeed [MAXSPEED]] [--minspeed [MINSPEED]]

Remote Driving

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to model h5 file. Model should be on the same
                        path.
  --image_folder [IMAGE_FOLDER]
                        Path to image folder. This is where the images from
                        the run will be saved.
  --maxspeed [MAXSPEED]
                        Maximum speed limit
  --minspeed [MINSPEED]
                        Minimum speed limit
```

## Result

<p align="center">
<img src="https://github.com/crypto-code/Udacity-Self-Driving-Car/blob/master/assets/result.gif" align="middle" />   </p>
