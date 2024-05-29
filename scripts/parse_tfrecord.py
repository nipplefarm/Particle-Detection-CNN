import tensorflow as tf

def load_dataset(tfrecord_file):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/class/label': tf.io.VarLenFeature(tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_png(features['image/encoded'], channels=3)
        image = tf.image.resize(image, [256, 256])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.sparse.to_dense(features['image/class/label'])[0]  # Ensure label is a scalar
        return image, label

    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset

def main():
    train_tfrecord = 'data/tfrecords/train.tfrecord'
    val_tfrecord = 'data/tfrecords/val.tfrecord'

    train_dataset = load_dataset(train_tfrecord)
    val_dataset = load_dataset(val_tfrecord)

    # Example to print the first element from the dataset
    for image, label in train_dataset.take(1):
        print(image.numpy(), label.numpy())

if __name__ == "__main__":
    main()
