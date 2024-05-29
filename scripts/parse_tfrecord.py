import tensorflow as tf

def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, [256, 256])  # Resize to 256x256
    image = image / 255.0  # Normalize to [0, 1]
    
    label = tf.cast(example['image/class/label'], tf.int32)
    
    return image, label

def load_dataset(tfrecords_files, batch_size=32):
    raw_dataset = tf.data.TFRecordDataset(tfrecords_files)
    dataset = raw_dataset.map(parse_tfrecord_fn)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    train_dataset = load_dataset(['data/train.tfrecord'])
    val_dataset = load_dataset(['data/val.tfrecord'])
    print("Datasets loaded successfully")
