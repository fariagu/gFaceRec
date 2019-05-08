# CREDIT: https://github.com/ageitgey/

import sys
import dlib
import cv2
import openface

import utils

def detect_and_crop(src_dir, dst_dir, file_name, percentage):
	# You can download the required pre-trained face detection model here:
	# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	predictor_model = "./shape_predictor_68_face_landmarks.dat"

	# print("Cropping image " + file_name)

	file_path = src_dir + file_name

	# Create a HOG face detector using the built-in dlib class
	face_detector = dlib.get_frontal_face_detector()
	face_pose_predictor = dlib.shape_predictor(predictor_model)
	# face_aligner = openface.AlignDlib(predictor_model)

	# Load the image
	image = cv2.imread(file_path)

	# Run the HOG face detector on the image data
	detected_faces = face_detector(image, 1)

	# print("Found {} faces in the image file {}".format(len(detected_faces), file_path))

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

		cv2.imwrite(dst_dir + file_name, image[top:bottom, left:right])

def batch_detect_and_crop():
    identity_dict = filenames_and_labels()
    images = []
    for img in os.listdir(utils.og_images_dir):
    # for img in os.listdir("C:/datasets/CelebA/img_crop/"):
        if img in identity_dict.keys():
            if identity_dict[img] < utils.num_classes:
                images.append(img)

    crop_pctgs = [0, 5, 10, 15, 20, 25, 30]
    base_dir = 
    src_dir = "no_crop"

    p = Pool(utils.n_workers)

    for percentage in crop_pctgs:
        dst_dir = "crop_{pctg:02d}/".format(pctg=percentage)
        p.map(detect_and_crop, images)
        # p.map(detect_align.main, os.listdir(utils.og_images_dir))
        # p.map(detect_align.main, os.listdir("C:/datasets/CelebA/tmp/"))

    p.close()

    # load_vectors_into_disk()

    print("DONE")