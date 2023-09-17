import streamlit as st

import cv2
import math
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")


offset = 20
imgSize = 300
counter = 0
labels = ["I", "love", "you", "hello"]
folder = "data/I" # add new word


def realTimeFeed():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        imgOutput = img.copy()
        try:
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                imgCropShape = imgCrop.shape

                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(f"prediction:{prediction}, index:{index}")

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    print(f"prediction:{prediction}, index:{index}")

                cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("imgWhite", imgWhite)
        except():
            print("")
        stframe.image(img, channels="BGR")

        cv2.imshow("Image", imgOutput)
        # cv2.imshow("feed", frame)
        if cv2.waitKey(1) & 0XFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

st.title("Bond Video Chat")
realTimeFeed()

