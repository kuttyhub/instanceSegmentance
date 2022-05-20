from random import random
import cv2
import numpy as np


def showMaskImage(masks,img, detection_count):
	
	height, width, _ = img.shape
	black_image = np.zeros((height, width, 3), np.uint8)
	black_image[:] = (100, 100, 0)

	for i in range(detection_count):
		box = boxes[0, 0, i]
		class_id = box[1]
		score = box[2]
		if score < 0.5:
			continue

		# Get box Coordinates
		x = int(box[3] * width)
		y = int(box[4] * height)
		x2 = int(box[5] * width)
		y2 = int(box[6] * height)

		roi = black_image[y: y2, x: x2]
		roi_height, roi_width, _ = roi.shape

		# Get the mask
		mask = masks[i, int(class_id)]
		mask = cv2.resize(mask, (roi_width, roi_height))
		_, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)


		# Get mask coordinates
		contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		color = randomColor()
		for cnt in contours:
			cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
	
	cv2.imshow("Black image", black_image)


def showOuputBoundedImage(img,detection_count):
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

	# Load image
	# img = cv2.imread("horse.jpg")
	img = cv2.imread("road.jpg")
	# img = cv2.imread("pexels.jpg")
	# img = cv2.imread("construction.jpg")


	# Detect objects
	blob = cv2.dnn.blobFromImage(img, swapRB=True)
	net.setInput(blob)

	boxes, masks = net.forward(["detection_out_final", "detection_masks"])
	# boxes= net.forward("detection_out_final")
	detection_count = boxes.shape[2]

	showMaskImage(masks,img,detection_count)
	showOuputBoundedImage(img,detection_count)



	cv2.waitKey(0)
