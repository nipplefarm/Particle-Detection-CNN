import os
import time
import tensorflow as tf
from evaluate_model import evaluate_and_plot
from tensorflow.keras import layers, models #type:ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard #type:ignore
import matplotlib.pyplot as plt
import pickle
import argparse
import kerastuner as kt
import numpy as np
from datetime import datetime
import visualkeras
from PIL import ImageFont
from parse_tfrecord import load_dataset

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

plot_directory = 'data/plots'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

start_time = time.time()

train_tfrecord_path = 'data/tfrecords/train.tfrecord'
val_tfrecord_path = 'data/tfrecords/val.tfrecord'
train_dataset = load_dataset(train_tfrecord_path).batch(16)
val_dataset = load_dataset(val_tfrecord_path).batch(8)

def model_builder(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.Int('conv_1_units', min_value=32, max_value=128, step=32), (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(hp.Int('conv_2_units', min_value=32, max_value=128, step=32), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(hp.Int('conv_3_units', min_value=32, max_value=128, step=32), (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=64), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

parser = argparse.ArgumentParser(description='Train a CNN model.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
args = parser.parse_args()

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory=r'D:\wdm230\hyperband',
                     project_name='particle_detection')

stop_early = EarlyStopping(monitor='val_loss', patience=10)

tuner.search(train_dataset, epochs=500, validation_data=val_dataset)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.8, patience=1, min_lr=1E-8)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_dataset,
    epochs=args.epochs,
    validation_data=val_dataset,
    callbacks=[reduce_lr, early_stopping, tensorboard_callback]
)

model.save('data/saved_model/model.h5')

with open('data/training_history/history.pkl', 'wb') as file:
    pickle.dump(history.history, file)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

evaluate_and_plot('data/saved_model/model.h5', val_tfrecord_path, 'data/training_history/history.pkl', plot_directory)

font_path = "arial.ttf"
font = ImageFont.truetype(font_path, 18)

visualkeras.layered_view(model, to_file='data/plots/model_architecture.png', legend=True, font=font).show()
