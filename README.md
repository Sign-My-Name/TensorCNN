# SignMyName - ISL Training Scripts

![SignMyName Logo](images/logo.png) <!-- Optional: Add your project's logo -->

## Overview

**SignMyName** is an application designed to teach children the Israeli Sign Language (ISL) using computer vision. This repository contains the core scripts used to train the machine learning models that power the application. These models are trained to recognize ISL gestures, making it possible for children to learn sign language interactively.

This repository includes the main scripts required to preprocess the data, train the models, and evaluate their performance.

## Repository Structure

- **main.py**: The main script that orchestrates the entire training process, including data loading, preprocessing, model training, and evaluation.
- **images_augmentations.py**: Contains functions for augmenting images, such as resizing, rotating, and applying transformations to prepare them for training.
- **manipulate_images.py**: Script for manipulating the original images and saving them in a processed format for training purposes.
- **create_metadata.py**: Generates the metadata files needed for training, including paths to images, class encodings, and other relevant data.

## How It Works

### 1. **Image Manipulation and Augmentation**
The `manipulate_images.py` script manipulates the original dataset images and augments them using the functions defined in `images_augmentations.py`. This step ensures that the model is trained on a diverse set of images, improving its robustness.

### 2. **Metadata Creation**
The `create_metadata.py` script generates metadata files in CSV format. These files contain essential information, such as image paths, class encodings, and weights, which are used during training.

### 3. **Model Training**
The `main.py` script handles the training process. It:
- Loads and preprocesses the data.
- Sets up data generators for training, validation, and testing.
- Defines and compiles the model architecture (ResNet50 or other models as defined).
- Trains the model with the specified parameters and saves the trained model.
- Evaluates the model using confusion matrices and other metrics.

## Installation

To get started with the training scripts, ensure you have the following installed:

- Python 3.8+
- TensorFlow
- Scikit-learn
- Pandas
- Numpy
- Matplotlib
- Seaborn

You can install the required packages using:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
