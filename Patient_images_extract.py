import json
import os

import cv2
import shutil
from tqdm import tqdm

# MVOR dataset code for visualization needs to be imported
import MVOR.lib.visualize_groundtruth as vis

def images_with_patients(json_file, destination_folder):
    """

    :param json_file: Annotations file
    :param destination_folder: contain copy of patient images
    :return: json file with patient's image_id, image_path and bounding_box
    """
    # json_file = './MVOR/annotations/camma_mvor_2018.json'

    with open(json_file) as file:
        annotations = json.load(file)
        # anno_2d, anno3d, mv_paths, imid_to_path = vis.create_index(file)

    imid_path = {}
    for idx in annotations['images']:
        imid_path[idx['id']] = idx['file_name']

    # print(f"Length: {len(imid_path)}\n {imid_path}")

    patients = []
    for p in annotations['annotations']:
        if p['person_role'] == 'patient':
            patient_id = p['image_id']
            patients.append({"image_id":patient_id, "image_path": imid_path[patient_id], "bbox": p['bbox']})

    print(f"no. of Patients Images: {len(patients)}")

    # destination_folder = 'J:/ATOS/MVOR_Patients/'
    counter = 00
    for image_file in tqdm(patients):
        counter += 1
        shutil.copyfile('J:/ATOS/camma_mvor_dataset/'+image_file['image_path'], destination_folder+str(counter)+".png")
    print(f"Copying Files Completed...")
    with open('./patient_images.json', 'w') as file:
        json.dump(patients, file)
    return patients

def visualize_patient_bbox(patient_json, destination_folder):
    """

    :param patient_json: Annotations file
    :param destination_folder: copy Images with bounding box of patients
    :return: string
    """
    # with open(patient_json) as file:
    #     patient_file = json.load(file)
    counter = 0
    for patient_image in tqdm(patient_json):
        counter += 1
        image_path = patient_image['image_path']
        bbox = patient_image['bbox']
        image = cv2.imread('J:/ATOS/camma_mvor_dataset/'+image_path)
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+int(bbox[0])), int(bbox[3]+int(bbox[1]))), (255,0,0), 2)
        cv2.imwrite(destination_folder+str(patient_image['image_id'])+'.png', image)

    return 'Bounding Box Images saved to directory...'


def train_test_split(json_file, train_ratio=0.80):
    """

    :param json_file: Coco Annotations file
    :param train_ratio: ratio of train split
    :return: train data and test data
    """
    root = 'J:/ATOS/camma_mvor_dataset/'
    with open(json_file) as file:
        annotations  = json.load(file)
    camera_angles = {'cam1': [], 'cam2': [], 'cam3': []}
    for json_item in annotations:
        cam = json_item['image_path'].split('/')[1]
        camera_angles[cam].append(json_item['image_path'])

    cam1 = int(len(camera_angles['cam1']) * train_ratio)
    cam2 = int(len(camera_angles['cam2']) * train_ratio)
    cam3 = int(len(camera_angles['cam3']) * train_ratio)

    cam1_train = camera_angles['cam1'][:cam1]
    cam2_train = camera_angles['cam2'][:cam2]
    cam3_train = camera_angles['cam3'][:cam3]

    cam1_test = camera_angles['cam1'][cam1:]
    cam2_test = camera_angles['cam2'][cam2:]
    cam3_test = camera_angles['cam3'][cam3:]

    train = cam1_train + cam2_train + cam3_train
    test = cam1_test + cam2_test + cam3_test

    print(f"Train: {len(train)}, test: {len(test)}")

    for image in train:
        image_name = "-".join(image.split('/'))
        shutil.copyfile(root + image, 'J:/ATOS/MVOR_dataset_split/train/'+image_name)

    for image in test:
        image_name = "-".join(image.split('/'))
        shutil.copyfile(root + image, 'J:/ATOS/MVOR_dataset_split/test/'+image_name)

    return train, test

def create_annotation_files(json_file, dest=None):
    """
    this function convert coco anootations to yolo format
    :param json_file: Json file that contains annotations is coco format
    :param dest: folder where the Images and annotaions are seperated for both classes
    :return: None
    """
    root = 'J:/ATOS/camma_mvor_dataset/'
    dest = 'J:/ATOS/MVOR_Images_YoloAnnotations/'

    # Creating Directories
    os.mkdir(dest+'clinician')
    os.mkdir(dest+'patient')

    os.mkdir(dest + 'clinician/Images')
    os.mkdir(dest + 'patient/Images')

    os.mkdir(dest + 'clinician/Annotations')
    os.mkdir(dest + 'patient/Annotations')

    with open(json_file) as file:
        annotations = json.load(file)

    imid_path = {}
    for idx in annotations['images']:
        imid_path[idx['id']] = idx['file_name']

    patients = []
    image_ids_list_p = set()
    image_ids_list_c = set()
    for p in tqdm(annotations['annotations']):
        if p['person_role'] == 'patient':
            image_path = imid_path[p['image_id']]
            class_id = 1
            bbox = p['bbox']
            text = str(class_id) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
            image_name = "-".join(image_path.split('/'))
            if p['image_id'] not in image_ids_list_p:
                image_ids_list_p.add(p['image_id'])
                shutil.copyfile(root+image_path, dest+'patient/Images/'+image_name)
                with open('J:/ATOS/MVOR_Images_YoloAnnotations/patient/Annotations/'+str(image_name.split('.')[0])+'.txt', 'w') as file:
                    file.write(text)

            elif p['image_id'] in image_ids_list_p:
                # shutil.copyfile(root + image_path, dest + 'patient/Images/' + image_name)
                with open('J:/ATOS/MVOR_Images_YoloAnnotations/patient/Annotations/' + str(image_name.split('.')[0]) + '.txt', 'a') as file:
                    file.write('\n')
                    file.write(text)
        elif p['person_role'] == 'clinician':
            image_path = imid_path[p['image_id']]
            class_id = 0
            bbox = p['bbox']
            text = str(class_id) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
            image_name = "-".join(image_path.split('/'))
            if p['image_id'] not in image_ids_list_c:
                image_ids_list_c.add(p['image_id'])
                shutil.copyfile(root + image_path, dest + 'clinician/Images/' + image_name)
                with open('J:/ATOS/MVOR_Images_YoloAnnotations/clinician/Annotations/' + str(image_name.split('.')[0]) + '.txt','w') as file:
                    file.write(text)

            elif p['image_id'] in image_ids_list_c:
                # shutil.copyfile(root + image_path, dest + 'patient/Images/' + image_name)
                with open('J:/ATOS/MVOR_Images_YoloAnnotations/clinician/Annotations/' + str(image_name.split('.')[0]) + '.txt',
                          'a') as file:
                    file.write('\n')
                    file.write(text)

    print('Completed successfully!')

def create_annotation_files_updated(json_file, dest=None):
    """
    this function convert coco anootations to yolo format
    :param json_file: Json file that contains annotations is coco format
    :param dest: folder where the Images and annotaions are seperated for both classes
    :return: None
    """
    root = 'J:/ATOS/camma_mvor_dataset/'
    dest = 'J:/ATOS/datasetv3.0/'

    # Creating Directories
    os.mkdir(dest+'images')
    os.mkdir(dest+'labels')

    # os.mkdir(dest + 'clinician/Images')
    # os.mkdir(dest + 'patient/Images')

    # os.mkdir(dest + 'clinician/Annotations')
    # os.mkdir(dest + 'patient/Annotations')

    with open(json_file) as file:
        annotations = json.load(file)

    imid_path = {}
    for idx in annotations['images']:
        imid_path[idx['id']] = idx['file_name']

    patients = []
    image_ids_list_p = set()
    image_ids_list_c = set()
    for p in tqdm(annotations['annotations']):
        if p['person_role'] == 'patient':
            image_path = imid_path[p['image_id']]
            class_id = 1
            bbox = p['bbox']
            x = bbox[0] * (1./480)
            w = bbox[1] * (1./480)
            y = bbox[2] * (1./640)
            h = bbox[3] * (1./640)
            text = str(class_id) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
            image_name = "-".join(image_path.split('/'))
            if p['image_id'] not in image_ids_list_p:
                image_ids_list_p.add(p['image_id'])
                shutil.copyfile(root+image_path, dest+'images/'+image_name)
                with open('J:/ATOS/datasetv3.0/labels/'+str(image_name.split('.')[0])+'.txt', 'w') as file:
                    file.write(text)

            elif p['image_id'] in image_ids_list_p:
                # shutil.copyfile(root + image_path, dest + 'patient/Images/' + image_name)
                with open('J:/ATOS/datasetv3.0/labels/' + str(image_name.split('.')[0]) + '.txt', 'a') as file:
                    file.write('\n')
                    file.write(text)
        elif p['person_role'] == 'clinician':
            image_path = imid_path[p['image_id']]
            class_id = 0
            bbox = p['bbox']
            x = bbox[0] * (1. / 640)
            w = bbox[1] * (1. / 640)
            y = bbox[2] * (1. / 480)
            h = bbox[3] * (1. / 480)
            text = str(class_id) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
            image_name = "-".join(image_path.split('/'))
            if p['image_id'] not in image_ids_list_p:
                image_ids_list_c.add(p['image_id'])
                shutil.copyfile(root + image_path, dest + 'images/' + image_name)
                with open('J:/ATOS/datasetv3.0/labels/' + str(image_name.split('.')[0]) + '.txt','w') as file:
                    file.write(text)

            elif p['image_id'] in image_ids_list_p:
                # shutil.copyfile(root + image_path, dest + 'patient/Images/' + image_name)
                with open('J:/ATOS/datasetv3.0/labels/' + str(image_name.split('.')[0]) + '.txt',
                          'a') as file:
                    file.write('\n')
                    file.write(text)

    print('Completed successfully!')

def split_dataset(directory, train_ratio=0.80, test_ratio=0.10, val_ratio=0.10):
    """

    :param directory: Folder that contained All the Images and Annotations
    :param train_ratio: training data ratio, default 0.80
    :param test_ratio: test data ratio, default 0.10
    :param val_ratio: validation data ratio, default 0.10
    :return: None
    """
    image_dir = directory+'/Images/'
    label_dir = directory+'/Annotations/'

    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    # ratio
    images_train_num = len(images) * train_ratio
    labels_train_num = len(labels) * train_ratio

    images_test_num = len(images) * test_ratio
    labels_test_num = len(labels) * test_ratio

    images_val_num = len(images) * val_ratio
    labels_val_num = len(labels) * val_ratio

    images_train = images[:int(images_train_num)]
    labels_train = labels[:int(labels_train_num)]

    images_test = images[int(images_train_num):int(images_train_num+images_test_num)]
    labels_test = labels[int(labels_train_num):int(labels_train_num + labels_test_num)]

    images_val = images[int(images_train_num + images_test_num):]
    labels_val = labels[int(labels_train_num + labels_test_num):]

    print(f"Already Exist") if os.path.exists(directory + 'train') else os.mkdir(directory+'train')
    print(f"Already Exist") if os.path.exists(directory + 'train/Images') else os.mkdir(directory+'train/Images')
    print(f"Already Exist") if os.path.exists(directory + 'train/Annotations') else os.mkdir(directory + 'train/Annotations')

    print(f"Already Exist") if os.path.exists(directory + 'test') else os.mkdir(directory + 'test')
    print(f"Already Exist") if os.path.exists(directory + 'test/Images') else os.mkdir(directory + 'test/Images')
    print(f"Already Exist") if os.path.exists(directory + 'test/Annotations') else os.mkdir(directory + 'test/Annotations')

    print(f"Already Exist") if os.path.exists(directory + 'val') else os.mkdir(directory + 'val')
    print(f"Already Exist") if os.path.exists(directory + 'val/Images') else os.mkdir(directory + 'val/Images')
    print(f"Already Exist") if os.path.exists(directory + 'val/Annotations') else os.mkdir(directory + 'val/Annotations')

    print(f"Directories Created...")
    for files in tqdm(images_train):
        shutil.copyfile(image_dir+files, directory+'train/Images/'+files.split('/')[-1])

    for files in tqdm(labels_train):
        shutil.copyfile(label_dir+files, directory+'train/Annotations/'+files.split('/')[-1])

    print(f"Copied Training Data")
    for files in tqdm(images_test):
        shutil.copyfile(image_dir+files, directory+'test/Images/'+files.split('/')[-1])

    for files in tqdm(labels_test):
        shutil.copyfile(label_dir+files, directory+'test/Annotations/'+files.split('/')[-1])
    print(f"Copied Testing Data")
    for files in tqdm(images_val):
        shutil.copyfile(image_dir+files, directory+'val/Images/'+files.split('/')[-1])

    for files in tqdm(labels_val):
        shutil.copyfile(label_dir+files, directory+'val/Annotations/'+files.split('/')[-1])
    print(f"Copied Validation Data")
    print(f"Train Images: {len(os.listdir(directory+'train/Images/'))} Train Labels: {len(os.listdir(directory+'train/Annotations/'))},\n"
          f" Test Images: {len(os.listdir(directory+'test/Images/'))} Test Labels{len(os.listdir(directory+'test/Annotations/'))}\n,"
          f" Validation Images: {len(os.listdir(directory+'val/Images/'))} Validation Labels: {len(os.listdir(directory+'val/Annotations/'))}")
def select_number_of_images(directory, destination, number=0):
    """

    :param directory: folder that contain Images and Annotations
    :param destination: folder you want to store your selected Images and Annotations
    :param number: number of Images Selected
    :return: None
    """

    images = os.listdir(directory+'/Images/')
    labels = os.listdir(directory+'/Annotations/')

    os.mkdir(destination+'/Images')
    os.mkdir(destination+'/Annotations')
    for files in tqdm(images[:number]):
        shutil.copyfile(directory+'/Images/'+files, destination+'/Images/'+files)
    for files in tqdm(labels[:number]):
        shutil.copyfile(directory+'/Annotations/'+files, destination+'/Annotations/'+files)

def split_dataset_via_annotations(dir):
    files = os.listdir(dir+'/labels')
    counter=0

    os.mkdir('J:/ATOS/datasetv3.1/labels/')
    os.mkdir('J:/ATOS/datasetv3.1/images/')
    for file in tqdm(files):
        with open(dir+'/labels/'+str(file)) as label_file:
            label_list = label_file.readlines()
            for idx in label_list:
                if idx.split(' ')[0] == '1':
                    counter+=1
                    shutil.copyfile(dir+'/labels/'+str(file), 'J:/ATOS/datasetv3.1/labels/'+str(file))
                    shutil.copyfile(dir+'/images/' + str(file.split('.')[0]+'.png'), 'J:/ATOS/datasetv3.1/images/' + str(file.split('.')[0]+'.png'))
    return str(counter)
def visualize_patient_yolo_annoatations(folder):
    images = os.listdir(folder+'/images')
    labels = os.listdir(folder+'/labels')

    for idx in range(0, len(images)):
        image = cv2.imread(str(folder+'/images/'+images[idx]))
        with open(folder+'/labels/'+labels[idx]) as label:
            lines = [x.rstrip('\n') for x in label.readlines()]
            for line in lines:
                bbox = line.split(' ')[1:]
                x = float(bbox[0]) * 480
                y = float(bbox[1]) * 480
                w = float(bbox[2]) * 640
                h = float(bbox[3]) * 640
                print(f"X: {int(x)}, Y: {int(y)}, W: {int(w)}, H: {int(h)}")
                print(f"Image Validation: {len(image)}, {type(image)}")
                cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 255), 2)
            cv2.imshow('Image with bounding box', image)
            cv2.waitKey(5000)

if __name__ == '__main__':
    # patient_file = images_with_patients('./MVOR/annotations/camma_mvor_2018.json', 'J:/ATOS/MVOR_Patients/')
    # visualize_patient_bbox(patient_file, 'J:/ATOS/MVOR_Patients/bbox_vis/')

    # Create Annotations for all images of Patient and Clinician
    # create_annotation_files('./MVOR/annotations/camma_mvor_2018.json')

    # Selecting only 774 images from clinician
    # select_number_of_images('J:/ATOS/MVOR_Images_YoloAnnotations/Dataset/clinician', 'J:/ATOS/MVOR_Images_YoloAnnotations/Dataset/clinician774', 774)

    # Create Annotations for all images of Patient and Clinician
    # create_annotation_files_updated('./MVOR/annotations/camma_mvor_2018.json')

    # split_dataset_via_annotations('J:/ATOS/datasetv2.0')
    visualize_patient_yolo_annoatations('J:/ATOS/datasetv3.1')

    # directory should contain Images and Annotations folder in same directory
    # split_dataset('J:/ATOS/MVOR_Images_YoloAnnotations/Dataset/clinician774/', 0.80, 0.10, 0.10)
    # split_dataset('J:/ATOS/MVOR_Images_YoloAnnotations/Dataset/patient/', 0.80, 0.10, 0.10)


