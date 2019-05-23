# CREDIT: https://github.com/ageitgey/

import sys
import dlib
import openface
import cv2

def detect_and_crop(file_name, src_dir, dst_dir, percentage):
    # # testing on windows
    # file_path = src_dir + file_name
    # image = cv2.imread(file_path)
    # cv2.imwrite(dst_dir + file_name, image)


    # You can download the required pre-trained face detection model here:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    predictor_model = "./shape_predictor_68_face_landmarks.dat"

    # print("Cropping image " + file_name)

    file_path = src_dir + file_name

    # # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    # face_pose_predictor = dlib.shape_predictor(predictor_model)
    # face_aligner = openface.AlignDlib(predictor_model)

    # Load the image
    image = cv2.imread(file_path)

    # # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    # print("Found {} faces in the image file {}".format(len(detected_faces), file_path))

    biggest_face_area = 0

    final_top, final_bottom, final_left, final_right = -1, -1, -1, -1

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges
        # print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        height, width, channels = image.shape

        # margin is calculated as a percentage of the face's coordinates [0, 100]
        margin_x = (face_rect.right() - face_rect.left()) * percentage / 100
        margin_y = (face_rect.bottom() - face_rect.top()) * percentage / 100

        left = int(face_rect.left() - margin_x) if face_rect.left() >= margin_x else 0
        top = int(face_rect.top() - margin_y) if face_rect.top() >= margin_y else 0
        right = int(face_rect.right() + margin_x) if face_rect.right() + margin_x <= width else width
        bottom = int(face_rect.bottom() + margin_y) if face_rect.bottom() + margin_y <= height else height

        face_area = (right - left) * (bottom - top)

        if face_area > biggest_face_area:
            biggest_face_area = face_area
            final_top, final_bottom, final_left, final_right = top, bottom, left, right

    if biggest_face_area > 0:
        cv2.imwrite(dst_dir + file_name, image[final_top:final_bottom, final_left:final_right])
