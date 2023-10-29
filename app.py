#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections.abc import Iterable
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as pg
import time

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


screenWidth, screenHeight = pg.size()
topLeftSector = [(1/4)*screenWidth, 1/4*screenHeight]
print(topLeftSector)
topRightSector = [(3/4)*screenWidth, 1/4*screenHeight]
bottomLeftSector = [(1/4)*screenWidth, 3/4*screenHeight]
bottomRightSector = [(3/4)*screenWidth, 3/4*screenHeight]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()
    return args

def userInput(corner):
    if 'topLeft' in corner:
        input("Look at top left corner and press enter")
    elif 'topRight' in corner:
        input("Look at top right corner and press enter")
    elif 'bottomLeft' in corner:
        input("Look at bottom left corner and press enter")
    elif 'bottomRight' in corner:
        input("Look at bottom right corner and press enter")


class videoManager:
    def __init__(self):
        self.eyeCalibration = False
        self.handDatum = False
        self.clickToggleCount = 0
        self.sector = ['left', 'top']
        args = get_args()

        model_path = 'model/face_models/face_landmarker.task'

        cap_device = args.device
        cap_width = args.width
        cap_height = args.height

        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        self.use_brect = True

        self.videoFeed = cv.VideoCapture(cap_device)                                   # Start video feed
        self.videoFeed.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        self.videoFeed.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        mp_face_mesh = mp.solutions.face_mesh
        self.faceMesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in self.point_history_classifier_labels
            ]

        self.FPS = CvFpsCalc(buffer_len=10)

        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)

        self.finger_gesture_history = deque(maxlen=self.history_length)

        self.mode = 0

    def captureFrame(self):
        ret, image = self.videoFeed.read()
        if not ret:
            return False
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        cv2image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        timestamp = self.videoFeed.get(cv.CAP_PROP_POS_MSEC)
        return cv2image, debug_image, timestamp
    
    def processVideo(self):
        while True:
            fps = self.FPS.get()

            key = cv.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = self.select_mode(key, self.mode)

            image, debugImage, timestamp = self.captureFrame()
            if image is False:
                break
            else:
                results = self.hands.process(image)
                cv.imshow('Hand Gesture Recognition', image)
            if self.eyeCalibration == False:
                self.calibrateGaze()
            else:
                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        brect = self.calc_bounding_rect(debugImage, hand_landmarks)
                        landmark_list = self.calc_landmark_list(debugImage, hand_landmarks)
                        # print(landmark_list[8])
                        if self.handDatum == False:
                            time.sleep(1)
                            self.handDatum = landmark_list[8]
                        if self.sector[0] == 'left':
                            if self.sector[1] == 'top':
                                # print("topLeft")
                                deltaX = ((landmark_list[8][0]-self.handDatum[0])*3)+topLeftSector[0]
                                deltaY = ((landmark_list[8][1]-self.handDatum[1])*3)+topLeftSector[1]
                                # print(deltaX, deltaY)
                                pg.moveTo(deltaX, deltaY, duration=0.1)
                            elif self.sector[1] == 'bottom':
                                # print("bottomLeft")
                                deltaX = ((landmark_list[8][0]-self.handDatum[0])*3)+bottomLeftSector[0]
                                deltaY = ((landmark_list[8][1]-self.handDatum[1])*3)+bottomLeftSector[1]
                                pg.moveTo(deltaX, deltaY, duration=0.1)
                        elif self.sector[0] == 'right':
                            if self.sector[1] == 'top':
                                # print("topRight")
                                deltaX = ((landmark_list[8][0]-self.handDatum[0])*3)+topRightSector[0]
                                deltaY = ((landmark_list[8][1]-self.handDatum[1])*3)+topRightSector[1]
                                pg.moveTo(deltaX, deltaY, duration=0.1)
                            elif self.sector[1] == 'bottom':
                                # print("bottomRight")
                                deltaX = ((landmark_list[8][0]-self.handDatum[0])*3)+bottomRightSector[0]
                                deltaY = ((landmark_list[8][1]-self.handDatum[1])*3)+bottomRightSector[1]
                                pg.moveTo(deltaX, deltaY, duration=0.1)
                        pre_processed_landmark_list = self.pre_process_landmark(
                            landmark_list)
                        pre_processed_point_history_list = self.pre_process_point_history(
                            debugImage, self.point_history)
                        self.logging_csv(number, mode, pre_processed_landmark_list,
                                    pre_processed_point_history_list)

                        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                        if hand_sign_id == 2: 
                            self.point_history.append(landmark_list[8])
                        else:
                            self.point_history.append([0, 0])

                        finger_gesture_id = 0
                        point_history_len = len(pre_processed_point_history_list)
                        if point_history_len == (self.history_length * 2):
                            finger_gesture_id = self.point_history_classifier(
                                pre_processed_point_history_list)

                        self.finger_gesture_history.append(finger_gesture_id)
                        most_common_fg_id = Counter(
                            self.finger_gesture_history).most_common()
                        thumbLength = self.distance(landmark_list[4], landmark_list[3])
                        thumbDistance = self.distance(landmark_list[4], landmark_list[6])
                        if thumbDistance < 0.5*thumbLength:
                            self.clickToggleCount += 1
                        else:
                            if self.clickToggleCount > 2 and self.clickToggleCount < 20:
                                print("Click")
                                pg.click()
                            self.clickToggleCount = 0
                        print(self.clickToggleCount)
                        debugImage = self.draw_bounding_rect(self.use_brect, debugImage, brect)
                        debugImage = self.draw_landmarks(debugImage, landmark_list)
                        debugImage = self.draw_info_text(
                            debugImage,
                            brect,
                            handedness,
                            self.keypoint_classifier_labels[hand_sign_id],
                            self.point_history_classifier_labels[most_common_fg_id[0][0]],
                        )
                        image = debugImage
                        # self.handDatum = landmark_list[8]
                else:
                    print("No hand found: Eye tracking mode")
                    self.point_history.append([0, 0])
                    faceResults = self.faceMesh.process(image)
                    image, gazeCoordinates = self.gaze(image, faceResults.multi_face_landmarks[0])
                    if gazeCoordinates is not None:
                        screenX = (gazeCoordinates[0]-(self.eyeCalibration[1][0][0]+self.eyeCalibration[1][2][0])/2)*self.eyeCalibration[0][0]
                        screenY = (gazeCoordinates[1]-(self.eyeCalibration[1][0][1]+self.eyeCalibration[1][2][1])/2)*self.eyeCalibration[0][1]
                        if screenX > pg.size()[0]*0.5:
                            self.sector[0] = 'right'
                        else:
                            self.sector[0] = 'left'
                        if screenY > pg.size()[1]*0.3:
                            self.sector[1] = 'bottom'
                        else:
                            self.sector[1] = 'top'
        #     if faceResults.multi_face_landmarks is not None:
        #         # print(len(faceResults.multi_face_landmarks[0]))
        #         gaze(image, faceResults.multi_face_landmarks[0])

            debugImage = self.draw_point_history(debugImage, self.point_history)
            debugImage = self.draw_info(debugImage, fps, mode, number)

            cv.imshow('Hand Gesture Recognition', image)

        self.videoFeed.release()
        cv.destroyAllWindows()
    def distance(self, point1, point2):
        return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

    def calibrateGaze(self):
        print("Calibrating Eyes")
        userInput('topLeft')
        image, debugImage, timestamp = self.captureFrame()
        faceResults = self.faceMesh.process(image)
        image, topLeftCoordinates = self.gaze(image, faceResults.multi_face_landmarks[0])
        userInput('topRight')
        image, debugImage, timestamp = self.captureFrame()
        faceResults = self.faceMesh.process(image)
        image, topRightCoordinates = self.gaze(image, faceResults.multi_face_landmarks[0])
        userInput('bottomLeft')
        image, debugImage, timestamp = self.captureFrame()
        faceResults = self.faceMesh.process(image)
        image, bottomLeftCoordinates = self.gaze(image, faceResults.multi_face_landmarks[0])
        userInput('bottomRight')
        image, debugImage, timestamp = self.captureFrame()
        faceResults = self.faceMesh.process(image)
        image, bottomRightCoordinates = self.gaze(image, faceResults.multi_face_landmarks[0])
        screenSize = pg.size()
        screenWidth = screenSize[0]
        screenHeight = screenSize[1]
        pixelsPerXGaze = screenWidth / (0.5*(abs(topRightCoordinates[0] - topLeftCoordinates[0])+abs(bottomRightCoordinates[0] - bottomLeftCoordinates[0])))
        pixelsPerYGaze = screenHeight / (0.5*(abs(topLeftCoordinates[1] - bottomLeftCoordinates[1])+abs(topRightCoordinates[1] - bottomRightCoordinates[1])))
        pixelsPerGaze = [pixelsPerXGaze, pixelsPerYGaze]
        gazeCoordinates = [topLeftCoordinates, topRightCoordinates, bottomLeftCoordinates, bottomRightCoordinates]
        self.eyeCalibration = [pixelsPerGaze, gazeCoordinates]

    def gaze(self, frame, points):
        """
        The gaze function gets an image and face landmarks from mediapipe framework.
        The function draws the gaze direction into the frame.
        """
        relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
        relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)
        '''
        2D image points.
        relative takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y) format
        '''
        image_points = np.array([
            relative(points.landmark[4], frame.shape),    # Nose tip
            relative(points.landmark[152], frame.shape),  # Chin
            relative(points.landmark[263], frame.shape),  # Left eye left corner
            relative(points.landmark[33], frame.shape),   # Right eye right corner
            relative(points.landmark[287], frame.shape),  # Left Mouth corner
            relative(points.landmark[57], frame.shape)    # Right mouth corner
        ], dtype="double")
        '''
        2D image points.
        relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
        at (x,y,0) format
        '''
        image_points1 = np.array([
            relativeT(points.landmark[4], frame.shape),  # Nose tip
            relativeT(points.landmark[152], frame.shape),  # Chin
            relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
            relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
            relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
            relativeT(points.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),       # Nose tip
            (0, -63.6, -12.5),     # Chin
            (-43.3, 32.7, -26),    # Left eye left corner
            (43.3, 32.7, -26),     # Right eye right corner
            (-28.9, -28.9, -24.1), # Left Mouth corner
            (28.9, -28.9, -24.1)   # Right mouth corner
        ])
        '''
        3D model eye points
        The center of the eye ball
        '''
        Eye_ball_center_left = np.array([[29.05],[32.7],[-39.5]])

        '''
        camera matrix estimation
        '''
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

        # 2d pupil location
        left_pupil = relative(points.landmark[468], frame.shape)

        # Transformation between image point to world point
        _, transformation, _ = cv.estimateAffine3D(image_points1, model_points)  # image to world transformation
        gaze = None
        if transformation is not None:  # if estimateAffine3D secsseded
            # project pupil image point into 3d world point 
            pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
            # 3D gaze point (10 is arbitrary value denoting gaze distance)
            S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 10
            # Project a 3D gaze direction onto the image plane.
            (eye_pupil2D, _) = cv.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            # project 3D head pose into the image plane
            (head_pose, _) = cv.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            # correct gaze for head rotation
            gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)
            # Draw gaze line into screen
            p1 = (int(left_pupil[0]), int(left_pupil[1]))
            p2 = (int(gaze[0]), int(gaze[1]))
            cv.line(frame, p1, p2, (0, 0, 255), 2)
        return frame, gaze

    def select_mode(self, key, mode):
        '''Select mode by keyboard input: Used for creating ML training data set'''
        number = -1
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        if key == 104:  # h
            mode = 2
        return number, mode

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(
            itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list

    def pre_process_point_history(self, image, point_history):
        image_width, image_height = image.shape[1], image.shape[0]

        temp_point_history = copy.deepcopy(point_history)

        base_x, base_y = 0, 0
        for index, point in enumerate(temp_point_history):
            if index == 0:
                base_x, base_y = point[0], point[1]

            temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
            temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

        temp_point_history = list(
            itertools.chain.from_iterable(temp_point_history))

        return temp_point_history

    def logging_csv(self, number, mode, landmark_list, point_history_list):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            csv_path = 'model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
        if mode == 2 and (0 <= number <= 9):
            csv_path = 'model/point_history_classifier/point_history.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *point_history_list])
        return

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        for index, landmark in enumerate(landmark_point):
            if index == 0:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text, finger_gesture_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if finger_gesture_text != "":
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
        return image

    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            if 0 <= number <= 9:
                cv.putText(image, "NUM:" + str(number), (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        return image


if __name__ == '__main__':
    videoManager = videoManager()
    videoManager.processVideo()
