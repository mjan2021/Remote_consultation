import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T




from vision.references.detection.engine import train_one_epoch, evaluate
#import vision.references.detection.utils as utils



# import pixellib
# from pixellib.custom_train import instance_custom_training

# train_maskrcnn = instance_custom_training()
# train_maskrcnn.modelConfig(network_backbone="resnet101", num_classes=2, batch_size=4)
# train_maskrcnn.load_pretrained_model("./assets/mask_rcnn_coco.h5")
# train_maskrcnn.load_dataset("./assets/MVOR_dataset_train/")
# train_maskrcnn.train_model(num_epochs=300, augmentation=True, path_trained_models="mask_rcnn_models")

# creating masks
root_p = 'J:/ATOS/camma_mvor_dataset/'


def get_image_shape(image_path):
    img = cv2.imread(image_path)
    return img.shape

def create_patients_masks(json_file):
    with open(json_file) as file:
        # image_id, image_path, bbox
        image_bbox = json.load(file)

    for idx in tqdm(image_bbox):
        # img = cv2.imread(root+idx['image_path'], 0)
        image_path = idx['image_path']
        bbox = idx['bbox']
        shape = get_image_shape(root_p+image_path)
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])
        # print(f"Image Shape: {img.shape} mask Shape: {shape}")
        mask = np.zeros((shape[0], shape[1]), dtype=np.int)  # initialize mask
        mask[y:y+h, x:x+w] = 255
        # mask_image = img + mask
        cv2.imwrite('J:/ATOS/MVOR_Patients/patients_masks/'+"-".join(idx['image_path'].split('/')), mask)
    return 'Masks created for '+str(len(os.listdir('J:/ATOS/MVOR_Patients/patients_masks/')))+'Patients'

class PatientDataset(torch.utils.data.Dataset):
    def __init__(self, root=None, transforms=None, target_transform=None):
        self.root = root
        self.transforms = transforms
        self.target_transform = transforms
        self.imgs = list(sorted(os.listdir(r'J:\ATOS\MVOR_Patients/bbox_vis/')))
        self.masks = list(sorted(os.listdir(r'J:\ATOS\MVOR_Patients/patients_masks/')))

    def __getitem__(self, idx):
        img_path = os.path.join(r'J:\ATOS\MVOR_Patients/bbox_vis/', self.imgs[idx])
        mask_path = os.path.join(r'J:\ATOS\MVOR_Patients/patients_masks/', self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        # target

        if self.transforms is not None:
            img = self.transforms(img)
            target = self.target_transform(target)
        else:
            img = T.ToTensor()(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)




if __name__ == '__main__':
    # create_patients_masks('./patient_images.json')
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to('cuda')
    # print(model.eval())
    # dataset = PatientDataset(root=None, transforms=get_transform(train=True), target_transform=get_transform(train=True))
    dataset = PatientDataset(root=None, transforms=None, target_transform=None)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    # torch.cuda.empty_cache()
    images, targets = next(iter(data_loader))

    image_list = list(image for image in images)
    # print(len(image_list))
    # print(image_list[0].size())
    targets_list=[]
    for i in range(len(image_list)):
        d={}
        d['boxes'] = targets['boxes'][i]
        d['labels'] = targets['labels'][i]
        d['image_id'] = targets['image_id'][i]
        targets_list.append(d)
    # print(image_list)
    # print(targets_list)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs=2
    device='cuda'
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader, device=device)

    print('Done')
    #
    # output = model(image_list, targets_list)
    # print(output)

