import imutils
import json
import time
import cv2
from tkinter import filedialog


def make_template():
        print('Choose Template Image for Comparision .')
        file_path = filedialog.askopenfilename()
        frame = cv2.imread(file_path, 1)
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray

def compare_frames(avg=None):
        print('Choose Image to Compare with Template .')
        file_path = filedialog.askopenfilename()
        frame = cv2.imread(file_path, 1)
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        #if avg is not already set
        if avg is None:
                print("Starting background model...")
                avg = gray.copy().astype("float")

        #cv2.accumulateWeighted(gray, avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        cv2.imshow("frame 1 : ", frame)
        cv2.waitKey(500)
        cv2.imshow("template frame : ", avg)
        cv2.waitKey(500)
        cv2.imshow("Difference between images : ", frameDelta)
        cv2.waitKey(0)


avg = make_template()
compare_frames(avg)

