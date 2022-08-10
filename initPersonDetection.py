import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import glob
from tqdm import tqdm
import torch
# Model Initialization
ins = instanceSegmentation()
ins.load_model("assets/pointrend_resnet50.pkl")
target_classes = ins.select_target_classes(person=True)

path_to_dataset = 'J:/ATOS/camma_mvor_dataset/day1/**/color/*.png'
list_of_files = glob.glob(path_to_dataset)
print(f"Total Files: {len(list_of_files)}")

# Cleaning up files names
cleanName_list = []
for file in list_of_files:
    cleanName_list.append( "/".join(file.split('\\')))

print(f"Total CleanFiles: {len(cleanName_list)}")

for cleanFile in tqdm(cleanName_list):
    cleanFileName = "-".join(cleanFile.split('/')[3:]).split('.')[0]
    # print(f"File Name: {cleanFileName}")
    ins.segmentImage(cleanFile, segment_target_classes=target_classes ,show_bboxes=True, output_image_name="J:/ATOS/processedImages/output_image"+cleanFileName+".jpg")

