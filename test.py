import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as tf
import mediapipe as mp
print("mediapipe is installed and working")
print("TensorFlow version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("cvzone is installed and working!")
from importlib.metadata import version
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier(r"C:\Users\HP\Desktop\SLR\Models\keras_model.h5" , r"C:\Users\HP\Desktop\SLR\Models\labels.txt")
offset = 20
imgSize = 300
counter = 0

model_path = r"C:\Users\HP\Desktop\SLR\Models\keras_model.h5"

if os.path.exists(model_path):
    print("Model file found")
    classifier = Classifier(model_path, "Model/labels.txt")
else:
    print(f"Model file not found at {model_path}")

from importlib.metadata import version, PackageNotFoundError

try:
   
    cvzone_version = version("cvzone")            # Try to fetch the version of cvzone
    print("cvzone version:", cvzone_version)
except PackageNotFoundError:
    print("cvzone is not installed")
 

labels = ["Hello","No","Yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction , index = classifier.getPrediction(imgWhite, draw= False)

       
        cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  

        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)