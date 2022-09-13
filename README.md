# Project: Object Detection in an Urban Environment

## Project Overview:
In this project, we are creating a convolutional neural network (CNN) to detect and classify objects using the Waymo dataset. This dataset consists of images of urban environments containing annotated vehicles, pedestrians and cyclists. After successful training, the CNN should be able to reliably detect vehicles, pedestrians and cyclists in similar urban environments. A self-driving car must never collide with any of these objects since it could result in loss of life to the occupant of the other vehicle, the pedestrian or the cyclist. These objects are different from other objects in an urban environment, such as traffic signals, median dividers and buildings because vehicles, pedestrians and cyclists are generally in motion. Thus, it is important for the self-driving car to recognize these objects and track their motion relative to its own motion so that a collision will never occur.

## Set up:
The code that is found in this repository was developed and tested in the Udacity classroom workspace according to the project instructions that were provided. This workspace had all of the necessary libraries and data provided.

## Dataset Analysis:
In the classroom workspace, the data from the Waymo dataset has been organized into three directories:
 - train: contains 86 tfrecord files used as the training files. Each tfrecord file contains multiple images.
 - val: contains 10 tfrecord files used for validation.
 - test: contains 3 tfrecord files used for testing the trained model and creating inference videos.

The files in the test directory are:

segment-12012663867578114640_820_000_840_000_with_camera_labels.tfrecord
segment-1208303279778032257_1360_000_1380_000_with_camera_labels.tfrecord
segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord

The files in the train and val directories are similar.

The first step of the project was to do exploratory data analysis using the data available in the data directories mentioned above. The Explore Data Analysis.ipynb Jupyter notebook file. The first step was to implement the display_images function. 
