# Particle Classificatioon Project

This project trains a Convolutional Neural Network (CNN) to classify things in images into three categories: particle, hole, and smear. The model is trained and evaluated using TensorFlow and leverages GPU acceleration for faster training.
This is mostly designed to check for particles on a nuclepore filter on scans done with an AFM.

## Directory Structure

- `data/`: Directory containing images, models, and tfrecords
- `scripts/`: Directory containing Python scripts for parsing data, training the model, and evaluating the model.
- `requirements.txt`: List of dependencies required for the project.
- `README.md`: This file.

## Steps to Run the Project
1. Annotate images using your preferred annotation software and export in Pascal VOC XML format.
2. In create_tfrecord, adjust features to your annotations and adjust classes/class map, do the same for the parse_tfrecord. Mkae sure to adjust file paths in all python files.
3. Create the .tfrecord
4. In train_model, adjust image augmentation, layers, and other settings as needed.
5. The model is saved and can be served using software of choice.
