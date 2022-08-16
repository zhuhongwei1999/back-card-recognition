# Bank Card Recognition Software

*The 8th "China Software Cup" Competition - Bank Card Number Recognition System*

> Author: Yiwu Cai, Hongwei Zhu and Kunpeng Ning

### System Implementation

#### Basic Functions

* Implemented card number area positioning, card number character recognition;
* Built a fast, easy and user-friendly working interface.

#### Additional Features

* Provided cross-platform multi-version packages;
* Supported space recognition between characters;
* Collected and enhanced data from complex and multiple scenes;
* Implemented auto error correction based on prior rule.

### Key Points

#### Model selection

Use YOLO-v3 network as the basic framework for card number area positioning and character recognition, increase the size of the network input layer and improve the resolution of the image.

#### Non-Maximum Suppression (NMS)

When predicting number bounding boxes, multiple prediction frames for a single number may occur. We added NMS between different classes based on original ones.

#### Data Collection and Augmentation

Due to the privacy of bank cards, only 350 bank card numbers were collected and tagged in real time with limited resources, but the training data was expanded to 140,000 copies through random rotation, panning, background replacement, highlight overlay, brightness/color/blur adjustment...

#### Auto Correction

The first six digits of the bank card number represent the card properties and its issuing bank, so there are limited combinations of numbers. We created a dictionary from bank card numbers to their properties, and implemented auto error correction function by string matching algorithm

#### Space recognition

The list of number box spacing is processed by Softmax, and if it is greater than the specified threshold, We determined there should be a space between two bounding boxes.

Here's a small [demo video](https://drive.google.com/file/d/1NfTTzmIYM9i9jYs_TWnN1L-RBFhurp6E/view?usp=sharing) for anyone interested. 
