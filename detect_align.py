# CREDIT: https://github.com/ageitgey/

import cv2
import dlib
# import openface

import utils

def main(file_name):
	# You can download the required pre-trained face detection model here:
	# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # predictor_model = "./shape_predictor_68_face_landmarks.dat"

    print("Cropping image " + file_name)

    # Take the image file name from the command line
    # file_name = sys.argv[1]

    file_name = utils.og_images_dir + file_name

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    # face_pose_predictor = dlib.shape_predictor(predictor_model)
    # face_aligner = openface.AlignDlib(predictor_model)

    # Take the image file name from the command line
    # file_name = sys.argv[1]

    # Load the image
    image = cv2.imread(file_name)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    # print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

    # Loop through each face we found in the image
    for face_rect in detected_faces:

        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        # print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Get the the face's pose
        # pose_landmarks = face_pose_predictor(image, face_rect)
        height, width, _ = image.shape

        # margin is calculated as a percentage of the face's coordinates [0, 100]
        margin_x = (face_rect.right() - face_rect.left()) * utils.margin_percentage / 100
        margin_y = (face_rect.bottom() - face_rect.top()) * utils.margin_percentage / 100

        left = int(face_rect.left() - margin_x) if face_rect.left() >= margin_x else 0
        top = int(face_rect.top() - margin_y) if face_rect.top() >= margin_y else 0
        right = int(face_rect.right() + margin_x) if face_rect.right() + margin_x <= width else width
        bottom = int(face_rect.bottom() + margin_y) if face_rect.bottom() + margin_y <= height else height

        # fr = dlib.drectangle(left, top, right, bottom)

        # Use openface to calculate and perform the face alignment
        # alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        # Save the aligned image to a file
        # cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)
        # cv2.imwrite(utils.crop_dir + file_name.split("/")[-1], image)

        # print(utils.crop_dir + file_name.split("/")[-1])

        cv2.imwrite(utils.crop_dir + file_name.split("/")[-1], image[top:bottom, left:right])

# main("/home/gustavoduartefaria/datasets/CelebA/img_align_celeba/000001.jpg")
