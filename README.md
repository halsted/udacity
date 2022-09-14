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

The first step of the project was to do exploratory data analysis using the data available in the data directories mentioned above. The Explore Data Analysis.ipynb Jupyter notebook file. The first step was to implement the display_images function and display 10 images from the training dataset. Here are some example images from the running the display images function.

![image](https://user-images.githubusercontent.com/7365421/190018480-6bd0eb9d-d406-4638-a89b-992d0f2c7018.png)
![image](https://user-images.githubusercontent.com/7365421/190018527-b9620917-5d74-4b61-aaea-c089924d0962.png)
![image](https://user-images.githubusercontent.com/7365421/190018561-8f0b9e73-59ff-45ff-9809-0b057aab4093.png)
![image](https://user-images.githubusercontent.com/7365421/190018594-c6f7c53c-df0e-421c-9fe9-c502684dcd67.png)
![image](https://user-images.githubusercontent.com/7365421/190018622-b36d0007-c67d-41c6-9096-1372dc79c6c5.png)
![image](https://user-images.githubusercontent.com/7365421/190018665-53bcb712-8529-4d1d-a364-ed0efea54600.png)
![image](https://user-images.githubusercontent.com/7365421/190018714-aa741d43-7216-46e2-850d-700b21b827b1.png)
![image](https://user-images.githubusercontent.com/7365421/190018729-af91bb46-7298-4e3f-ba0e-20d37474a3ea.png)
![image](https://user-images.githubusercontent.com/7365421/190018764-d1fb5e77-c6f9-4287-a645-886f38f3b5a0.png)
![image](https://user-images.githubusercontent.com/7365421/190018804-cf743ea8-17fe-4357-9063-1f29b0fa5831.png)

From the instructions in the Jupyter notebook, the object classes are required to have color coded bounding boxes with vehicles in red, pedestrians in blue and cyclists in green. As you can see from the images above, there are very few pedestrians or cyclists, but many cars. In fact, I had to run the script many times to search for images with a cyclist. A couple of images with cyclists and pedestrians are shown below. 

![image](https://user-images.githubusercontent.com/7365421/190019659-0b1c525e-6556-496f-88c2-3ca81f52f570.png)
![image](https://user-images.githubusercontent.com/7365421/190019850-d580d65c-75ff-49c9-b7aa-c1b066e919c0.png)
![image](https://user-images.githubusercontent.com/7365421/190019914-39ec6f24-e8d5-4c4e-bca5-55ef9058fa02.png)
![image](https://user-images.githubusercontent.com/7365421/190022702-a8462535-f94e-4e33-bd90-dbd007900639.png)
![image](https://user-images.githubusercontent.com/7365421/190022757-7a3a1994-dca5-4ee9-a2c5-fceba7658b43.png)


The observation that there were so few pedestrians and cyclists in the dataset images led me to write some code to count the relative frequency of vehicles, pedestrians and cyclists in a random set of images in the training dataset. The output from this code led to results such as:

![image](https://user-images.githubusercontent.com/7365421/190020940-b26765b6-7bfd-4837-ad88-46d0e2c9551f.png)
![image](https://user-images.githubusercontent.com/7365421/190021091-d62f3e6d-d3f4-4aec-89dd-75e857773e48.png)
![image](https://user-images.githubusercontent.com/7365421/190021206-372ba1a4-9766-4416-a9de-490607c340cf.png)

As can be seen from these statistics, there are indeed very few cyclists in the dataset images. There are far more vehicles than pedestrians and far more pedestrians than cyclists. From the data above, there are about 3.5 more vehicles than cyclists and about 38.9 times more pedestrians than cyclists. There are very few cyclists indeed and this could be a weakness in the dataset.

As part of the additional exploratory data analysis, I displayed 5 images from the val folder to compare validation images with training images. Here are two images containing a cyclist from the validation data. However, the blue bounding boxes seem to be incorrect (false positive) annotations of pedestrians.

![image](https://user-images.githubusercontent.com/7365421/190022857-5bb81933-cb13-4343-b506-a017dbf9e502.png)
![image](https://user-images.githubusercontent.com/7365421/190022109-4bf94bf3-e489-4610-b9f4-6341763e7206.png)

## Training

***Reference experiment***

For my reference experiment, I ran the default pipeline.config, which is the config for a SSD (Single Shot MultiBox Detector) Resnet 50 640x640 model in Tensorflow. This config file uses the Tensorflow Object Detection API. I followed the instructions given in the Project Instructions page exactly. The results I got were quite confusing so I posted my results and questions to the Udacity forum (https://knowledge.udacity.com/questions/897885). Here is the result of my training from Tensorboard:

![image](https://user-images.githubusercontent.com/7365421/190024551-6d8e42f3-23a7-4e2d-971f-4a98eb2f8d6a.png)

As you can see from the figures, the total loss at the end of the training was around 23. The mentor (Chaeseong L) told me that the loss was too high so the training was not successful. I tried it more than once with the default files, but got similar results. The mentor mentioned that in order to be successful, the total loss should be under 2.0. 

***Improve on the reference***

I made many iterations in order to improve the training to reach a total loss of under 2.0. I tried to run two different  models (faster_rcnn_resnet101_v1_640x640_coco17_tpu-8 and ssd_resnet101_v1_fpn_640x640_coco17_tpu-8), but neither of these would run successfully in the classroom workspace environment. It seems some of the needed libraries were missing. 

I also tried adding data augmentations based on suggestions from mentor Chaeseong L. In the end, I updated the config file to have a batch_size of 8 and added several data augmentations: random_adjust_contrast (min_delta: 0.6, max_delta: 1.0), random_rgb_to_gray (probability: 0.2), and random_adjust_brightness (max_delta: 0.2). I experimented with the Explore augmentations.ipynb notebook and thought that these data augmentations would provide good diversity to the dataset images. I was able to get the training total loss to less than 2.0 as shown in the Tensorboard figures below.

![image](https://user-images.githubusercontent.com/7365421/190026720-f5a69890-368e-44dd-a71b-3a3e6e1fe497.png)

The images from the evaluation from Tensorboard seem to show that the CNN was detecting vehicles fairly well.

![image](https://user-images.githubusercontent.com/7365421/190026939-0e9c02e4-1994-4bfa-ac5e-e58bd76cbdf0.png)

However, the mAP seems to be quite low so it seems that something may still be going wrong.

![image](https://user-images.githubusercontent.com/7365421/190027139-f2b29f25-1013-421a-9f44-094ff7313425.png)
![image](https://user-images.githubusercontent.com/7365421/190027298-cb5b8734-4834-4ed3-bb58-8311148592c4.png)

I created the animation file and it does detect vehicles. An earlier animation that I did lacked bounding boxes, but for the latest model produced an animation that detected many vehicles. However, my animation.gif file is 186 MB and I cannot upload it to this github repo. When I try to upload it, I get the message that the file needs to be 25 MB or less.





  





