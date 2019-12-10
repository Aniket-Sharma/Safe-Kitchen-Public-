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
    #content = imutils.resize(content, width=min(400, content.shape[1]))
    cv2.imshow("Original Image : ", frame)
    cv2.waitKey(0)
    a1, a2, k1, k2, k3, k4 = 5, 20, 80, 150, 190, 255 
    height, width, _ = frame.shape
    print(frame.shape)
    smoke = 0 
    for i in range(int(height)):
        for j in range(int(width)):
            m = max(int(frame[i][j][0]), int(frame[i][j][1]), int(frame[i][j][2]))
            n = min(int(frame[i][j][0]), int(frame[i][j][1]), int(frame[i][j][2]))
            i = (int(frame[i][j][0])+int(frame[i][j][1])+int(frame[i][j][2])) / 3
            a = m-n
            if a <= a1 and (i>=k1 and i<=k2):
                frame[i][j] = [0, 255, 0]
                smoke+=1
            elif a<= a2 and (i>=k3 and i<=k4):
                frame[i][j] = [0,0,255]
                smoke+=1
            print(a, i)

    prob = smoke*100 / width*height
    
    print('Smoke Pixel percentage  : '+str(prob)+' % ')
    cv2.imshow("Smoke Pixels heighlited  : " , frame)
    if prob>25:
        cv2.imshow("Smoke Pixels heighlited  : " , frame)
        cv2.waitKey(500)
        return True
    return False

print(SIP_IP())

