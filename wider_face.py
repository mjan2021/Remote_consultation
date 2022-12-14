import tensorflow
import tensorflow_datasets as tfds
import glob
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
from lxml import etree
import math
import json
# Loading the dataset

Image_Files = glob.glob('assets/FDDB/FDDB-folds/*.txt')
# print(Image_Files)
annotationsFiles = []
imageFiles = []
for file in Image_Files:
    if file.split('-')[-1] == 'ellipseList.txt':
        cleanedFileName = '/'.join(file.split('\\'))
        annotationsFiles.append(cleanedFileName)
    else:
        cleanedFileName = '/'.join(file.split('\\'))
        imageFiles.append(cleanedFileName)

# print(f"Image Files: {imageFiles} \n Annotations File: {annotationsFiles}")
fileNames = []
for each_file in imageFiles:
    with open(each_file, 'r') as iFile:
        for each_line in iFile.readlines():
            # print(each_line)
        # fileName = '/'.join(iFile.readline().split('\n'))
            fileNames.append(each_line)

DIR_JPG = 'JPEGImages'
DIR_ANNOTAITON = 'Annotations'


def img2xml(path, objects, shape):
    if len(shape) != 3:
        return None
    root = etree.Element("annotation")
    folder = etree.SubElement(root, "folder")
    filename = etree.SubElement(root, "filename")
    source = etree.SubElement(root, "source")
    databases = etree.SubElement(source, "database")

    folder.text = "VOC2007"
    filename.text = str(path).zfill(6)
    databases.text = "FDDB"

    size = etree.SubElement(root, "size")
    width = etree.SubElement(size, "width")
    height = etree.SubElement(size, "height")
    depth = etree.SubElement(size, "depth")
    depth.text = str(shape[2])
    width.text = str(shape[1])
    height.text = str(shape[0])

    obj_count = 0
    for obj in objects:
        obj = [float(i) for i in obj.split()]
        # the smallest circumscribed parallelogram
        # https://github.com/nouiz/lisa_emotiw/blob/master/emotiw/common/datasets/faces/FDDB.py
        maj_rad = obj[0]
        min_rad = obj[1]
        angle = obj[2]
        xcenter = obj[3]
        ycenter = obj[4]
        cosin = math.cos(math.radians(-angle))
        sin = math.sin(math.radians(-angle))

        x1 = cosin * (-min_rad) - sin * (-maj_rad) + xcenter
        y1 = sin * (-min_rad) + cosin * (-maj_rad) + ycenter
        x2 = cosin * (min_rad) - sin * (-maj_rad) + xcenter
        y2 = sin * (min_rad) + cosin * (-maj_rad) + ycenter
        x3 = cosin * (min_rad) - sin * (maj_rad) + xcenter
        y3 = sin * (min_rad) + cosin * (maj_rad) + ycenter
        x4 = cosin * (-min_rad) - sin * (maj_rad) + xcenter
        y4 = sin * (-min_rad) + cosin * (maj_rad) + ycenter
        width = [x1, x2, x3, x4]
        height = [y1, y2, y3, y4]
        xmin = int(min(width))
        xmax = int(max(width))
        ymin = int(min(height))
        ymax = int(max(height))

        # check if out of box
        if(xmin > 0 and ymin > 0 and xmax < shape[1] and ymax < shape[0]):
            obj_count += 1
            object_ = etree.SubElement(root, "object")
            name = etree.SubElement(object_, "name")
            name.text = "face"
            pose = etree.SubElement(object_, "pose")
            pose.text = "Unspecified"
            truncated = etree.SubElement(object_, "truncated")
            truncated.text = "0"
            difficult = etree.SubElement(object_, "difficult")
            difficult.text = "0"
            # bounded box
            bndbox = etree.SubElement(object_, "bndbox")
            xmin_ = etree.SubElement(bndbox, "xmin")
            ymin_ = etree.SubElement(bndbox, "ymin")
            xmax_ = etree.SubElement(bndbox, "xmax")
            ymax_ = etree.SubElement(bndbox, "ymax")
            xmin_.text = str(xmin)
            ymin_.text = str(ymin)
            xmax_.text = str(xmax)
            ymax_.text = str(ymax)

    if obj_count > 0:
        et = etree.ElementTree(root)
        return et
    else:
        return None


destination_path = Path('assets/FDDB/FDDB_2010')
# create annotation folder
annotation_output_folder = destination_path / 'Annotations'
annotation_output_folder.mkdir(parents=True, exist_ok=True)
# create jpge folder
jpge_image_output_folder = destination_path / 'JPEGImages'
jpge_image_output_folder.mkdir(parents=True, exist_ok=True)
# print(f"FileName: {fileNames}")
original_pics_folds = Path('assets/FDDB/originalPics')
for each_file in annotationsFiles:
    with open(each_file, 'r') as f:
        image_with_target = [i.strip() for i in f]

output_file_id = len(glob.glob(str(jpge_image_output_folder / '*.jpg')))
current_index = 0
annotationJson = []
while (current_index < len(image_with_target)):
    """
    example1: 2 object in 2002/08/02/big/img_769
    2002/08/02/big/img_760
    2
    58.887348 37.286244 1.441974 88.083450 78.409537  1
    60.381076 40.303691 1.377522 260.502940 102.769525  1
    example2: 1 object in 2002/08/07/big/img_1453
    2002/08/02/big/img_760
    1
    67.995400 38.216200 -1.559920 208.966471 109.764400  1
    2002/08/07/big/img_1453
    """
    src_image_file = '%s.jpg' % image_with_target[current_index]
    path_img = str(original_pics_folds / src_image_file)
    shape = np.array(Image.open(path_img)).shape
    current_index += 1
    len_obj = int(image_with_target[current_index])
    current_index += 1
    objects = image_with_target[current_index: current_index + len_obj]
    current_index += len_obj
    image_id = str(output_file_id).zfill(6)

    print(f"id: {output_file_id}, objects: {objects}, shape: {shape}")
    annotationJson.append({"id": output_file_id, "objects": objects, "shape": shape})

    etree_object = img2xml(output_file_id, objects, shape)
    if etree_object:
        xml_output_name = "%s.xml" % image_id
        xml_output_path = str(annotation_output_folder / xml_output_name)
        etree_object.write(xml_output_path, pretty_print=True)
        jpge_output_name = "%s.jpg" % image_id
        jpge_output_path = str(jpge_image_output_folder / jpge_output_name)
        shutil.copy2(path_img, jpge_output_path)
        output_file_id += 1

with open('FDDB_annotations.json', 'w') as fdd_json:
    fdd_json.write(str(annotationJson))