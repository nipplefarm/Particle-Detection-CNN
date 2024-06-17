import os
import xml.etree.ElementTree as ET

def remove_labels(xml_file, labels_to_remove):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in labels_to_remove:
            root.remove(obj)
    tree.write(xml_file)

def process_annotations(annotation_dir, labels_to_remove):
    for xml_file in os.listdir(annotation_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotation_dir, xml_file)
            remove_labels(xml_path, labels_to_remove)
            print(f"Processed {xml_file}")

# Directory containing the Pascal VOC XML annotations
annotation_dir = r'C:\Users\wdm230\Desktop\particle-detection-cnn\data\annotated_images\Annotations'
# Labels to remove
labels_to_remove = ['Hole', 'Smear']

process_annotations(annotation_dir, labels_to_remove)
