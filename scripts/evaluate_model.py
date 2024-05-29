import tensorflow as tf
import matplotlib.pyplot as plt
from parse_tfrecord import load_dataset

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load datasets
val_dataset = load_dataset(['data/val.tfrecord'])

# Evaluate the model
test_loss, test_acc = model.evaluate(val_dataset)
print('Test accuracy:', test_acc)

# Plot training & validation accuracy/loss values
history = model.history.history
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
