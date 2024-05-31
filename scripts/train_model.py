import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from parse_tfrecord import load_dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import pickle
import argparse
from evaluate_model import evaluate_and_plot


# Argument parser
parser = argparse.ArgumentParser(description='Train a CNN model.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
args = parser.parse_args()

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

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_image(image, label):
    image = tf.numpy_function(datagen.random_transform, [image], tf.float32)
    return image, label

# Load datasets
train_dataset = load_dataset('data/tfrecords/train.tfrecord').map(augment_image).batch(32).shuffle(1000)
val_dataset = load_dataset('data/tfrecords/val.tfrecord').batch(32)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),  # Add Dropout for regularization
    layers.BatchNormalization(),  # Add Batch Normalization
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=.1,
    patience=25,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=50
)

# Train the model with early stopping
history = model.fit(
    train_dataset,
    epochs=args.epochs,  # Use the number of epochs from the command line argument
    validation_data=val_dataset,
    callbacks=[reduce_lr, early_stopping]
)

# Save the model
model.save('data/saved_model/model.h5')

# Save the training history
with open('data/training_history/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

evaluate_and_plot('data/saved_model/model.h5', 'data/tfrecords/val.tfrecord', 'data/training_history/history.pkl')