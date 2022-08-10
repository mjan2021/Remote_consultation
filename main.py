# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import mediapipe as mp

filetype = 'C:/Users/tanve/PycharmProjects/Telehealth_ATOS/assets/face_image.jpg'
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def face_detection(filetype, type=''):
    if type == 'image':
        cv2.imshow('window', cv2.imread(filetype))
        # For static images:
        IMAGE_FILES = [filetype]
        with mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:
            for idx, file in enumerate(IMAGE_FILES):
                image = cv2.imread(file)
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Draw face detections of each face.
                if not results.detections:
                    continue
                annotated_image = image.copy()
                for detection in results.detections:
                    print('Nose tip:')
                    print(mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.LEFT_EYE))
                    mp_drawing.draw_detection(annotated_image, detection)
                cv2.imshow('Annotated Image', annotated_image)
                cv2.waitKey(0)

    elif type == 'video':
        cap = cv2.VideoCapture(filetype)
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        print(mp_face_detection.get_key_point(
                            detection, mp_face_detection.FaceKeyPoint.LEFT_EYE))
                        mp_drawing.draw_detection(image, detection)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

def face_mesh(video_path):
    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(video_path)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    print(f"Face_larndmarks: {results.multi_face_landmarks}")
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # face_detection('C:/Users/tanve/PycharmProjects/Telehealth_ATOS/assets/face_image.jpg', type='image')
    # face_detection('Z:/VIDEOS/1003 1004/Video/2021-11-09/T000301000000.asf', type='video')
    # face_mesh('Z:/VIDEOS/1003 1004/Video/2021-11-09/T000301000000.asf')
    face_detection('J:/ATOS/camma_mvor_dataset/day1/cam1/color/000013.png', type='image')
