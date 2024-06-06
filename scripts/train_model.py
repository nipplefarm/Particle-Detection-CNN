import tensorflow as tf
from tensorflow.keras import layers, models # type:ignore
from parse_tfrecord import load_dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau # type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type:ignore
import pickle
import argparse
from evaluate_model import evaluate_and_plot
import kerastuner as kt

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

tuner.search(train_dataset, epochs=500 , validation_data=val_dataset)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first conv layer is {best_hps.get('conv_1_units')}.
""")

# Build the model with the optimal hyperparameters and train it
model = tuner.hypermodel.build(best_hps)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=3, min_lr=1E-5)

history = model.fit(
    train_dataset,
    epochs=args.epochs,  # Use the number of epochs from the command line argument
    validation_data=val_dataset,
    callbacks=[reduce_lr]
)

# Save the model
model.save('data/saved_model/model.h5')

# Save the training history
with open('data/training_history/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

evaluate_and_plot('data/saved_model/model.h5', 'data/tfrecords/val.tfrecord', 'data/training_history/history.pkl')
