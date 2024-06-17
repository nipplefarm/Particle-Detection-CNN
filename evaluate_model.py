import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os
from parse_tfrecord import load_dataset

def evaluate_and_plot(model_path, val_tfrecord_path, history_path, plot_directory, batch_size=32):
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    
    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})

    # Load validation dataset
    val_dataset = load_dataset(val_tfrecord_path).batch(batch_size)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(val_dataset)
    print('Test accuracy:', test_acc)

    # Load training history
    with open(history_path, 'rb') as file:
        history = pickle.load(file)

    # Plot training & validation accuracy/loss values
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.ylim(0, 1.1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.ylim(bottom=0)
    plt.title('Training and validation loss')
    plt.legend()

    # Save the plot
    plot_path = os.path.join(plot_directory, 'training_history.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

# Example usage
plot_directory = 'data/plots'
evaluate_and_plot('data/saved_model/model.h5', 'data/tfrecords/val.tfrecord', 'data/training_history/history.pkl', plot_directory)
