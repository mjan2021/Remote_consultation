from torchvision import transforms as T
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import os
from tqdm import tqdm
import glob
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def pose_detection_keypointrcnn(image):
    # create a model object from the keypointrcnn_resnet50_fpn class
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    # call the eval() method to prepare the model for inference mode.
    model.eval()

    # create the list of keypoints.
    keypoints = ['nose','left_eye','right_eye', 'left_ear','right_ear','left_shoulder',
                 'right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip',
                 'right_hip','left_knee', 'right_knee', 'left_ankle','right_ankle']



    # Read the image using opencv
    image_name = image
    img_path = image
    img = cv2.imread(img_path)

    # preprocess the input image
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img)

    # forward-pass the model
    # the input is a list, hence the output will also be a list
    output = model([img_tensor])[0]


    def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
        # initialize a set of colors from the rainbow spectrum
        cmap = plt.get_cmap('rainbow')
        # create a copy of the image
        img_copy = img.copy()
        # pick a set of N color-ids from the spectrum
        color_id = np.arange(1,255, 255//5).tolist()[::-1]
        # color_id = np.arange(1,255,255).tolist()[::-1]
        # iterate for every person detected
        for person_id in range(len(all_keypoints)):
          # check the confidence score of the detected person
          if confs[person_id]>conf_threshold:
            # grab the keypoint-locations for the detected person
            keypoints = all_keypoints[person_id, ...]
            # grab the keypoint-scores for the keypoints
            scores = all_scores[person_id, ...]
            # iterate for every keypoint-score
            for kp in range(len(scores)):
                # check the confidence score of detected keypoint
                if scores[kp]>keypoint_threshold:
                    # convert the keypoint float-array to a python-list of integers
                    keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                    # pick the color at the specific color-id
                    color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                    # draw a circle over the keypoint location
                    cv2.circle(img_copy, keypoint, 2, color, 1)

        return img_copy

    keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=2)

    # cv2.imshow("Keypoints", keypoints_img)
    # cv2.waitKey(0)

    def get_limbs_from_keypoints(keypoints):
      limbs = [
            [keypoints.index('right_eye'), keypoints.index('nose')],
            [keypoints.index('right_eye'), keypoints.index('right_ear')],
            [keypoints.index('left_eye'), keypoints.index('nose')],
            [keypoints.index('left_eye'), keypoints.index('left_ear')],
            [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
            [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
            [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
            [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
            [keypoints.index('right_hip'), keypoints.index('right_knee')],
            [keypoints.index('right_knee'), keypoints.index('right_ankle')],
            [keypoints.index('left_hip'), keypoints.index('left_knee')],
            [keypoints.index('left_knee'), keypoints.index('left_ankle')],
            [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
            [keypoints.index('right_hip'), keypoints.index('left_hip')],
            [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
            [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
            ]
      return limbs

    limbs = get_limbs_from_keypoints(keypoints)


    def draw_skeleton_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
        # initialize a set of colors from the rainbow spectrum
        cmap = plt.get_cmap('rainbow')
        # create a copy of the image
        img_copy = img.copy()
        # check if the keypoints are detected
        if len(output["keypoints"]) > 0:
            # pick a set of N color-ids from the spectrum
            colors = np.arange(1, 255, 255 // len(all_keypoints)).tolist()[::-1]

            # iterate for every person detected
            for person_id in range(len(all_keypoints)):
                # check the confidence score of the detected person
                if confs[person_id] > conf_threshold:
                    # grab the keypoint-locations for the detected person
                    keypoints = all_keypoints[person_id, ...]

                    # iterate for every limb
                    for limb_id in range(len(limbs)):
                        # pick the start-point of the limb
                        limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
                        # pick the start-point of the limb
                        limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
                        # consider limb-confidence score as the minimum keypoint score among the two keypoint scores
                        limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
                        # check if limb-score is greater than threshold
                        if limb_score > keypoint_threshold:
                            # pick the color at a specific color-id
                            color = tuple(np.asarray(cmap(colors[person_id])[:-1]) * 255)
                            # draw the line for the limb
                            cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), color, 2)

        return img_copy

    # overlay the skeleton in the detected person
    skeletal_img = draw_skeleton_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"],keypoint_threshold=2)

    # cv2.imshow("Pose Detection", skeletal_img)
    image_name = image_name.split('/')[-1]
    cv2.imwrite('J:/ATOS/MVOR_Patients/pose_detection_keypoint/'+str(image_name), skeletal_img
                )
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return "Process Completed..."

def pose_detection_mediapipe(image_list):
    print(f"Running function..")

    # For static images:
    IMAGE_FILES = image_list
    BG_COLOR = (192, 192, 192)  # gray
    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        for idx, file in tqdm(enumerate(IMAGE_FILES)):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            # print(
            #     f'Nose coordinates: ('
            #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            #     f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            # )

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imwrite('J:/ATOS/MVOR_Patients/pose_detection/' + str(idx) + '.png', annotated_image)

            # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(
            #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    return "Process Completed...."


if __name__ == "__main__":
    # pose_detection_keypointrcnn("J:/ATOS/MVOR_Patients/33.png")
    images = glob.glob('J:/ATOS/MVOR_Patients/*.png')
    for idx in images:
        img = "/".join(idx.split('\\'))
        pose_detection_keypointrcnn(img)
    # print(image_list)
    # print(f"pose detection meidapipe")
    # images = glob.glob('J:/ATOS/MVOR_Patients/*.png')
    # image_list = []
    # for idx in images:
    #     image_list.append("/".join(idx.split('\\')))
    # print(image_list)
    # pose_detection_mediapipe(image_list)
