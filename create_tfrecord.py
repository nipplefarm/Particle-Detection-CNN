import os
import glob
import xml.etree.ElementTree as ET
import tensorflow as tf
from sklearn.model_selection import train_test_split

def parse_yolo_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    objects = []
    for member in root.findall('object'):
        value = {
            'class': member.find('name').text.lower(),  # Normalize to lowercase
            'xmin': int(member.find('bndbox/xmin').text),
            'xmax': int(member.find('bndbox/xmax').text),
            'ymin': int(member.find('bndbox/ymin').text),
            'ymax': int(member.find('bndbox/ymax').text)
        }
        objects.append(value)
    
    return objects

def create_tf_feature(image_file, annotations):
    image_string = open(image_file, 'rb').read()
    image_shape = tf.image.decode_png(image_string).shape
    
    height = image_shape[0]
    width = image_shape[1]
    
    filename = os.path.basename(image_file).encode('utf8')
    image_format = b'png'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    class_map = {'particle': 0, 'hole': 1, 'smear': 2}
    
    for annotation in annotations:
        xmins.append(annotation['xmin'] / width)
        xmaxs.append(annotation['xmax'] / width)
        ymins.append(annotation['ymin'] / height)
        ymaxs.append(annotation['ymax'] / height)
        classes_text.append(annotation['class'].encode('utf8'))
        classes.append(class_map[annotation['class']])
    
    tf_feature = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
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

def create_tfrecord(image_files, annotation_dir, tfrecord_file):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for image_file in image_files:
            try:
                # Construct the corresponding annotation file path
                xml_file = os.path.join(annotation_dir, os.path.basename(image_file).replace('.png', '.xml'))
                annotations = parse_yolo_xml(xml_file)
                tf_feature = create_tf_feature(image_file, annotations)
                writer.write(tf_feature.SerializeToString())
                print(f"Processed {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

def main():
    img_dir = 'data/annotated_images/images'
    annotation_dir = 'data/annotated_images/Annotations'
    output_dir = 'data/tfrecords'
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(img_dir, '*.png'))
    print(f"Found {len(image_files)} image files.")  # Debug information
    
    if len(image_files) == 0:
        print("No image files found. Please check the directory path and file extensions.")
        return
    
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    print(f"Creating TFRecord for training set with {len(train_files)} images.")
    create_tfrecord(train_files, annotation_dir, os.path.join(output_dir, 'train.tfrecord'))
    
    print(f"Creating TFRecord for validation set with {len(val_files)} images.")
    create_tfrecord(val_files, annotation_dir, os.path.join(output_dir, 'val.tfrecord'))
    
    print(f'TFRecord files created successfully: {len(train_files)} training images, {len(val_files)} validation images')

if __name__ == "__main__":
    main()
