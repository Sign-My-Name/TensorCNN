# SignMyName - ISL Training Scripts

![SignMyName Logo](images/logo.png)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**SignMyName** is an app that helps children learn the Israeli Sign Language (ISL) through computer vision. This repository contains the scripts used to train the machine learning models that recognize ISL gestures, making learning more interactive and fun.

This repo includes the key scripts to preprocess data, train models, and evaluate their performance.

## Repository Structure

- **main.py**: The main script that manages the entire training process, including data loading, preprocessing, model training, and evaluation.
- **images_augmentations.py**: Functions for augmenting images, such as resizing, rotating, and applying other transformations to prepare them for training.
- **manipulate_images.py**: Script for processing the original images and saving them in a format ready for training.
- **create_metadata.py**: Generates metadata files needed for training, including paths to images, class encodings, and other relevant details.

## How It Works

### 1. **Image Manipulation and Augmentation**
The `manipulate_images.py` script processes and augments the original dataset images using functions defined in `images_augmentations.py`. This step ensures the model trains on a diverse set of images, improving accuracy.

### 2. **Metadata Creation**
The `create_metadata.py` script generates metadata files in CSV format. These files contain essential information, such as image paths, class encodings, and weights, used during training.

### 3. **Model Training**
The `main.py` script handles the entire training process. It:
- Loads and preprocesses the data.
- Sets up data generators for training, validation, and testing.
- Defines and compiles the model architecture (e.g., ResNet50).
- Trains the model with the given parameters and saves the trained model.
- Evaluates the model using metrics like confusion matrices and accuracy.

## Installation

To get started, clone the repository and install the required packages listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
