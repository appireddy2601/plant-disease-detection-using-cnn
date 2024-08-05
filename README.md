# Plant Disease Detection Using CNN

## Overview
This repository contains a project focused on detecting plant diseases using Convolutional Neural Networks (CNNs). The goal of this project is to accurately classify and identify various plant diseases from images of plant leaves.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Installation](#installation)
9. [License](#license)

## Introduction
Plant diseases can significantly impact agricultural productivity. Early detection and classification of these diseases can help in timely intervention and treatment, thereby reducing crop loss. This project leverages CNNs to classify images of plant leaves into different disease categories.

## Dataset
The dataset used for training and testing the model consists of images of healthy and diseased plant leaves. The dataset is publicly available and contains images labeled with various disease categories. You can download the dataset from [this link](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

## Model Architecture
The CNN model used in this project is designed to effectively capture the features of plant leaves and classify them accurately. The architecture includes multiple convolutional layers, pooling layers, and fully connected layers. Below is a brief overview of the architecture:
- Input layer: 128x128 RGB image
- Convolutional layers: 3 layers with ReLU activation
- Pooling layers: 2 layers with max pooling
- Fully connected layers: 2 layers with dropout
- Output layer: Softmax activation for multi-class classification

## Training
The model is trained using the following parameters:
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Batch size: 32
- Number of epochs: 50
- Validation split: 20%

## Evaluation
The model is evaluated on a separate test set to assess its performance. Key metrics such as accuracy, precision, recall, and F1-score are calculated to measure the effectiveness of the model.

## Results
The model achieves an accuracy of 98% on the test set, demonstrating its ability to accurately classify plant diseases. Below are some sample predictions:

| Image | True Label | Predicted Label |
|-------|------------|-----------------|
| ![Sample Image 1](#) | Healthy | Healthy |
| ![Sample Image 2](#) | Diseased | Diseased |

## Installation
To run this project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git

## License

Feel free to adjust the sections and content as per your specific project requirements. Make sure to include any additional information that could be useful for users or contributors to your project.
