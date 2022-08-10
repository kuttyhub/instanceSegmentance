from random import random
import cv2
import numpy as np


def showOuputBoundedImage(img,detection_count,boxes):
	height, width, _ = img.shape
	for i in range(detection_count):
		box = boxes[0, 0, i]
		score = box[2]
		if score < 0.5:
			continue

		# Get box Coordinates
		x = int(box[3] * width)
		y = int(box[4] * height)
		x2 = int(box[5] * width)
		y2 = int(box[6] * height)

		color = randomColor()
		cv2.rectangle(img, (x, y), (x2, y2), color, 3)
	cv2.imshow("Image", img)

def randomColor():
	return (random() * 255, random() * 255, random() * 255)

	
if __name__ == '__main__':
# Loading Mask RCNN
	net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
										"dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

	images = ['pexels1.jpg','pexels2.jpg','pexels3.jpg','pexels4.jpg','pexels5.jpg','construction.jpg']

	for image in images:
		img = cv2.imread(image)

		# Detect objects
		blob = cv2.dnn.blobFromImage(img, swapRB=True)
		net.setInput(blob)

		boxes= net.forward("detection_out_final")
		detection_count = boxes.shape[2]

		showOuputBoundedImage(img,detection_count,boxes)

		cv2.waitKey(0)
