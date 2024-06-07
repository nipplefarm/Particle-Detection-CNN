import os
import time  # Import the time module
import tensorflow as tf
from tensorflow.keras import layers, models # type:ignore
from parse_tfrecord import load_dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau # type:ignore
import matplotlib.pyplot as plt
import pickle
import argparse
from evaluate_model import evaluate_and_plot
import kerastuner as kt
import numpy as np
import random

# Ensure TensorFlow uses the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

# Start the timer
start_time = time.time()

# List of possible augmentations
def apply_rotation(image):
    return tf.image.rot90(image, k=random.randint(1, 3))

def apply_width_shift(image):
    return tf.image.random_crop(image, size=tf.shape(image))

def apply_height_shift(image):
    return tf.image.random_crop(image, size=tf.shape(image))

def apply_shear(image):
    return tf.image.central_crop(image, central_fraction=0.8)

def apply_zoom(image):
    return tf.image.random_crop(image, size=tf.shape(image))

def apply_horizontal_flip(image):
    return tf.image.random_flip_left_right(image)

augmentations = [apply_rotation, apply_width_shift, apply_height_shift, apply_shear, apply_zoom, apply_horizontal_flip]

def random_augment(image, label):
    original_shape = tf.shape(image)
    chosen_augmentations = random.sample(augmentations, 3)
    for aug in chosen_augmentations:
        image = aug(image)
    image = tf.image.resize(image, original_shape[:2])
    return image, label

def augment_image_5_times(image, label):
    augmented_images = [random_augment(image, label) for _ in range(5)]
    augmented_images, augmented_labels = zip(*augmented_images)  # Separate images and labels
    return tf.data.Dataset.from_tensor_slices((list(augmented_images), list(augmented_labels)))

def visualize_augmentations(dataset, num_images=5):
    plt.figure(figsize=(15, 15))
    for i, (image, label) in enumerate(dataset.take(num_images)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(np.clip(image.numpy(), 0, 1))  # Ensure the values are between 0 and 1
        plt.title(f"Label: {label.numpy()}")
        plt.axis("off")
    plt.show()

# Load datasets
train_dataset = load_dataset('data/tfrecords/train.tfrecord')
val_dataset = load_dataset('data/tfrecords/val.tfrecord').batch(32)

# Apply the custom augmentation function
train_dataset_augmented = train_dataset.flat_map(augment_image_5_times)

# Visualize augmented images
visualize_augmentations(train_dataset_augmented, num_images=5)

# Combine datasets
combined_train_dataset = train_dataset.concatenate(train_dataset_augmented).batch(16).shuffle(1000)

# Hyperparameter tuning
def model_builder(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('conv_1_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('conv_2_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('conv_3_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('conv_4_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hp.Int('conv_5_units', min_value=32, max_value=256, step=32), (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('dense_units', min_value=256, max_value=1024, step=128), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))  # Add Dropout for regularization
    model.add(layers.BatchNormalization())  # Add Batch Normalization
    model.add(layers.Dense(hp.Int('dense_2_units', min_value=256, max_value=1024, step=128), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Argument parser
parser = argparse.ArgumentParser(description='Train a CNN model.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
args = parser.parse_args()

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='data/hyperband',
                     project_name='particle_detection')

# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(combined_train_dataset, epochs=500, validation_data=val_dataset)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first conv layer is {best_hps.get('conv_1_units')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1E-4)

history = model.fit(
    combined_train_dataset,
    epochs=args.epochs,
    validation_data=val_dataset,
    callbacks=[reduce_lr]
)

# Save the model
model.save('data/saved_model/model.h5')

# Save the training history
with open('data/training_history/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

evaluate_and_plot('data/saved_model/model.h5', 'data/tfrecords/val.tfrecord', 'data/training_history/history.pkl')

# End the timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
