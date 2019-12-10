# Imports the Google Cloud client library and streaming libraries
import io
import os
import cv2
import math

import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *

from google.cloud import vision
from google.cloud.vision import types

import numpy as np
import tensorflow as tf

def SIP_IP():
    print('You are in option to detect smoke in an image.')
    file_path = filedialog.askopenfilename()
    frame = cv2.imread(file_path)
    frame = imutils.resize(frame, width=min(300, frame.shape[1]))
    cv2.imshow("Original Image : ", frame)
    cv2.waitKey(10)
    print("sdfdsf")
    a1, a2, k1, k2, k3, k4 = 5, 20, 80, 150, 190, 255 
    height, width, _ = frame.shape
    print(frame.shape)
    smoke = 0
    for i in range(width):
        for j in range(height):
            m = max(frame[i][j][0] ,frame[i][j][1], frame[i][j][2])
            n = min(frame[i][j][0] ,frame[i][j][1], frame[i][j][2])
            a=m-n
            intensity = (frame[i][j][0] +frame[i][j][1]+ frame[i][j][2])/3
            if (a <= a2 and a>=a1) and ((intensity>=k1 and intensity<=k2) or (intensity>=k3 and intensity<=k4)):
                frame[i][j] = [0,255,0]
                smoke+=1

    prob = (smoke*100) / (  width*height)
    print(smoke)
    print('Smoke Pixel percentage  : '+str(prob)+' % ')
    cv2.imshow("Smoke Pixels heighlited  : " , frame)
    cv2.waitKey(0)
    if prob>25:
        cv2.imshow("Smoke Pixels heighlited  : " , frame)
        cv2.waitKey(0)
        return True
    return False

SIP_IP()

