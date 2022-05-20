import cv2
import numpy as np
import os
 
# Get the list of all images
path = os.getcwd() + "//images"
dir_list = os.listdir(path)

# Loading Mask RCNN
net = cv2.dnn.readNetFromTensorflow("cnn/frozen_inference_graph_coco.pb",
									"cnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Generate random colors
colors = np.random.randint(0, 255, (80, 3))

for image in dir_list:

	img = cv2.imread("images/" + image)
	height, width, _ = img.shape

	# Detect objects
	blob = cv2.dnn.blobFromImage(img, swapRB=True)
	net.setInput(blob)

	boxes, masks = net.forward(["detection_out_final", "detection_masks"])
	detection_count = boxes.shape[2]

	for i in range(detection_count):
		box = boxes[0, 0, i]
		class_id = box[1]
		score = box[2]

		# for skiping ojects with less than 0.5 score
		if score <= 0.5:
			continue

		# Get box Coordinates.
		x = int(box[3] * width)
		y = int(box[4] * height)
		x2 = int(box[5] * width)
		y2 = int(box[6] * height)

		# Get Color for the class id.
		color = colors[int(class_id)]
		color = (int(color[0]), int(color[1]), int(color[2]))

		# Draw Rectangle for the object.
		cv2.rectangle(img, (x, y), (x2, y2), color, 2)


		roi = img[y: y2, x: x2]
		roi_height, roi_width, _ = roi.shape

		# Get the mask
		mask = masks[i, int(class_id)]
		mask = cv2.resize(mask, (roi_width, roi_height))
		_, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

		# Get mask coordinates
		contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			cv2.fillPoly(roi, [cnt], color)

	# Display Image.
	cv2.imshow("Image", img)
	cv2.waitKey(0)