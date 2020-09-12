# IMPORT PACKAGES
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
################################################

################################################
def detect_mask(frame, face_Net, mask_Net):
	# Take Dimensions so as to contruct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

	# pass blob to CNN to predict output
	face_Net.setInput(blob)
	detections = face_Net.forward()
	print(detections.shape)

	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		#get probability of the detection (also known as confidence)
		confidence = detections[0, 0, i, 2]

		# Having threshold for detections to filter out weaker ones
		if confidence > 0.5:
			#Get (x, y)-coordinates of boundidng boxes
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Add the face and bounding boxes
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Will detect only one face
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = mask_Net.predict(faces, batch_size=32)

	# return tuple of location and prediction of face
	return (locs, preds)

##################### LOAD FACE DETECTOR
prototxtpath = r"face_detector\deploy.prototxt"
weightspath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face_Net = cv2.dnn.readNet(prototxtpath, weightspath)

################### LOAD MASK DETECTOR
mask_Net = load_model("mask_detector.model")

# Initialize Video Stream
vs = VideoStream(src=0).start()

# Loop over Video Frames
while True:
	# Grab the frame and resize it to 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect mask
	(locs, preds) = detect_mask(frame, face_Net, mask_Net)

	# Loop over detected face
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Color used to draw bounding box
		#Green for Mask, Red for no mask
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Probability included
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		#Display output and probability
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()