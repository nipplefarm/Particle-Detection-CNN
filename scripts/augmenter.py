import os
import glob
import tensorflow as tf
import imgaug.augmenters as iaa
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def read_voc_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'Particle':
            bbox = obj.find('bndbox')
            x_min = int(bbox.find('xmin').text)
            y_min = int(bbox.find('ymin').text)
            x_max = int(bbox.find('xmax').text)
            y_max = int(bbox.find('ymax').text)
            annotations.append(BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max))
    return annotations

def load_image_and_annotations(image_path, annotation_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3).numpy()
    bboxes = read_voc_annotations(annotation_path)
    return image, bboxes

def create_tf_feature(image, annotations, filename):
    image_string = tf.io.encode_jpeg(tf.convert_to_tensor(image)).numpy()
    height, width, _ = image.shape
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    class_map = {'particle': 0}
    
    for annotation in annotations:
        xmins.append(annotation.x1 / width)
        xmaxs.append(annotation.x2 / width)
        ymins.append(annotation.y1 / height)
        ymaxs.append(annotation.y2 / height)
        classes_text.append(b'particle')
        classes.append(class_map['particle'])
    
    tf_feature = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    }))
    
    return tf_feature

def create_augmented_dataset(image_dir, annotation_dir, train_output_path, val_output_path, num_augmentations=10, val_split=0.2):
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    train_files, val_files = train_test_split(image_files, test_size=val_split, random_state=42)
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20)),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0, 1.0))
    ])
    
    with tf.io.TFRecordWriter(train_output_path) as train_writer, tf.io.TFRecordWriter(val_output_path) as val_writer:
        for files, writer in [(train_files, train_writer), (val_files, val_writer)]:
            for image_file in files:
                annotation_file = os.path.splitext(os.path.basename(image_file))[0] + '.xml'
                annotation_path = os.path.join(annotation_dir, annotation_file)
                if os.path.exists(annotation_path):
                    image, bboxes = load_image_and_annotations(image_file, annotation_path)
                    bboxes_on_image = BoundingBoxesOnImage(bboxes, shape=image.shape)
                    for i in range(num_augmentations):
                        augmented_image, augmented_bboxes = seq(image=image, bounding_boxes=bboxes_on_image)
                        augmented_bboxes = augmented_bboxes.clip_out_of_image().remove_out_of_image_fraction(0.5)
                        augmented_filename = f"{os.path.splitext(os.path.basename(image_file))[0]}_aug_{i}.png"
                        tf_feature = create_tf_feature(augmented_image, augmented_bboxes.bounding_boxes, augmented_filename)
                        writer.write(tf_feature.SerializeToString())
                        print(f"Processed {augmented_filename}")

image_dir = 'data/annotated_images/images'
annotation_dir = 'data/annotated_images/Annotations'
train_output_path = 'data/tfrecords/train.tfrecord'
val_output_path = 'data/tfrecords/val.tfrecord'
create_augmented_dataset(image_dir, annotation_dir, train_output_path, val_output_path)
