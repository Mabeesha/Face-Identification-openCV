import os
import numpy as np
from PIL import Image
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0 #Ids that are used to label y_labels
label_ids = {} #dictionary that saves the labels and the IDs.
y_labels = []
x_train = []
z = 0

#saving the images in x_train and y_labels
for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "_").lower()

			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			ID = label_ids[label]
			#print(label_ids)
			#y_labels.append(label) # some number
			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
			pil_image = Image.open(path).convert("L") # grayscale
			#size = (550, 550)
			#final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(pil_image, "uint8")

			#img = "img" + str(z)  + ".png"
			#cv2.imwrite(img, image_array)
			#z += 1
			#print(image_array)
			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(ID)

				#img = "img" + str(z)  + ".png"
				#cv2.imwrite(img, roi)
				#z += 1
print(label_ids)
print(y_labels)

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")