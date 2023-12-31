## Implementation:
In our project, we implemented a custom feature extractor model based on Convolutional
Neural Network (CNN), specifically designed to track vehicles in real-time traffic scenarios. This
model combined with the kalman filter gives accurate prediction of vehicles in the next state.
The CNN model was meticulously trained on a diverse dataset comprising various classes of
vehicles, ensuring robust detection across different types and sizes of vehicles. This training
enabled the model to accurately identify vehicles in video streams, even in challenging
conditions.

## Diagram:
![Alt text](media/diagram1.png)

A Pre-trained yolov4 model is used to give detections of vehicle class. These detections are
passed to our custom feature extractor CNN model and Kalman filter. The appearance vector
information from the feature extractor model is combined with the motion information from the
Kalman filter to give better prediction of objects in the next state. A cosine distance metric is
used for similarity calculation. The Hungarian algorithm calculates intersection over union (IOU)
for association between our prediction and tracklet.

## Code
Model Architecture 
![Alt text](media/image.png)


## Commands to run - 

### save yolov4 model
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4 --model yolov4
### Run yolov4 deep sort object tracker on video
python object_tracker.py --weights ./checkpoints/yolov4 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/cars.avi
### Run yolov4 deep sort object tracker with custom trained feature model on video
python object_tracker.py --weights ./checkpoints/yolov4 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/cars.avi --custom_feature_extractor_model

### save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny
### Run yolov4-tiny object tracker on video.
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/cars_tiny.avi --tiny
### Run yolov4-tiny object tracker with custom trained feature model on video.
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/cars_tiny.avi --tiny --custom_feature_extractor_model

## Access weights and dataset
weights and dataset - https://drive.google.com/drive/folders/1lWpMoyE733gAqb6XMG_MNQGOdcaKD79s?usp=drive_link

Feature Extraction model Training – 
Dataset comprising of about 184 different classes of cars.
Total training images are about 18,400 cropped images of cars.


## Output - 
![Alt text](outputs/image.png)

video demo - [![Video Demo](outputs/cars.jpg)](outputs/cars.avi)




# INSTALLATION
## yolov4-deepsort
Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.


## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading Official YOLOv4 Pre-trained Weights
Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights


## Resulting Video
As mentioned above, the resulting video will save to wherever you set the ``--output`` command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the ``--output_format`` flag, by default it is set to AVI codec which is XVID.


## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
```

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
