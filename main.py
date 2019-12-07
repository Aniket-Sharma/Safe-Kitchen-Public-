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

# Opencv pre-trained SVM with HOG people features 

# HOGCV = cv2.HOGDescriptor()
# HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class Alarms :
    def Fire():
        print('ALERT ! There is a fire in the kitchen. Please Check it out. ')
    def Fall():
        print('ALERT ! The person has fallen. Please check it out. ')
    def Object():
        print('ALERT ! There is something unsual on the floor, watch your step . ')


def person_detector(image):
    print('checking if there\'s someone in the room ............')
    # retruns the reactangle array highlithing the location of person in given image.  
    image = imutils.resize(image, width=min(400, image.shape[1]))
    clone = image.copy()
    (rects, weights) = HOGCV.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    if len(result)==0 :
        print('There\'s no person in the image. ')
    else:
        print('There are '+str(len(result))+' persons in the image.')
    return result

# detecting fire in a frame using image processing 
def FIF_IP(frame):
    #returns True or False based on whether there's fire or not. It will also output live frame after and before processing.   
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
    lower = [18, 50, 50]
    upper = [35, 255, 255]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(hsv, lower, upper)
    
    output = cv2.bitwise_and(frame, hsv, mask=mask)
    no_red = cv2.countNonZero(mask)

    prob_of_fire = (int(no_red) * 100)/40000  

    cv2.putText(img=output, text='Chances of fire : '+str(prob_of_fire)+' % ' , org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0))
    
    # cv2.imshow("Live Video : ", frame)

    # cv2.imshow("Probabilty of fire : ", output)
    cv2.imshow("Output" , output)
    # cv2.putText(img, prob_of_fire, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    #print("output:", frame)
    #print(int(no_red))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False

    if int(no_red) > 20000:
        print ('Fire detected in this image. ')
        return True

    print('There are '+str(prob_of_fire)+' %  chances of fire in this frame. ')

    return False


def FIF_ML(frame):
    # retrun True or False based on whether the ML algo detects fire in image or not. 
    return True

def fire_in_image():

    print('You are in option to detect fire in an image.')
    file_path = filedialog.askopenfilename()
    content = cv2.imread(file_path, 0)
    cv2.imshow("Live Video : ", content)
    # content = np.float32(content)
    # content.astype(np.float32).dtype
    root = tk.Tk()
    root.title('Fire Detection App')
    root.geometry("720x580")
    leftFrame = Frame(root)
    label2 = Label(leftFrame, text="Detecting fire in an Image. ",fg="Red",  font=('Verdana', 25, 'bold'))
    label2.pack()

    label0 = Label(leftFrame, text="Please wait while the system looks for signs of fire in given image. ", fg="Green", font=('Verdana', 15, 'bold'))
    label0.pack()
    
    cv2.putText(img=content, text='checking if there\'s someone in the room ............' , org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
    cv2.imshow("Live Video : ", content)

    persons_in_room = person_detector(content)
    # persons_in_room = person_detector_temp(frame)
    cv2.putText(img=content, text='Number of persons in room : '+str(len(persons_in_room)), org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 0))
    cv2.imshow("Live Video : ", content)

    if (FIF_IP(content)):

        label4 = Label(leftFrame, text="The Image Processing algorithm shows positive signs of Fire.\n lets run the ML algo.  ", fg="Green", font=('Verdana', 15, 'bold'))
        label4.pack()

        if (FIF_ML(content)):
            label5 = Label(leftFrame, text="The ML algorithm also shows positive signs of Fire.\n There is fire in given image.", fg="Green", font=('Verdana', 15, 'bold'))
            label5.pack()
            Alarms.Fire()
            return
        else:
            label5 = Label(leftFrame, text="The ML algorithm also shows negative signs of Fire.\n The Image Processing algo created a false alarm .", fg="Green", font=('Verdana', 15, 'bold'))
            label5.pack()
            return
        
    label6 = Label(leftFrame, text="The Image Processing algorithm shows negative signs of Fire.\n There is no fire in given image.  ", fg="Green", font=('Verdana', 15, 'bold'))
    label6.pack()

    return
    
def fire_in_video():
    print('You are in option to detect fire in a video.')
    video_path = filedialog.askopenfilename()
    # with io.open(video_path, 'rb') as video_file:
    video = cv2.VideoCapture(video_path)
    # fps = math.floor(vs.get(cv2.CAP_PROP_FPS))
    ret_val = True
    writer = 0
    frame_count = 0 
    while True:
        (grabbed, frame) = video.read()

        if not grabbed:
            break

        frame_count+=1
        # will use the algorithm on every 100th frame

        if frame_count>=4:

            cv2.putText(img=frame, text='checking if there\'s someone in the room ............' , org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            cv2.imshow("Live Video : ", frame)

            # print('checking if there\'s someone in the room ............')
            persons_in_room = person_detector(frame)
            # persons_in_room = person_detector_temp(frame)
            cv2.putText(img=frame, text='Number of persons in room : '+str(len(persons_in_room)), org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 0))
            cv2.imshow("Live Video : ", frame)

            if (len(persons_in_room)==0):
                print('checking for signs of fire in frame .............. ')
                cv2.putText(img=frame, text='There\'s no one in the room so checking for signs of fire in frame .............. ', org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 0))
                cv2.imshow("Live Video : ", frame)
                
                if (FIF_IP(frame)):
                    Frames_with_Fire.append(frame)
                    print('Detected fire in '+str(len(Frames_with_Fire))+' Consequent Frames. ')
                    cv2.putText(img=frame, text='Detected fire in '+str(len(Frames_with_Fire))+' Consequent Frames. ', org=(10, 55), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 0))
                    cv2.imshow("Live Video : ", frame)
                
                    if (len(Frames_with_Fire) >= 10 ):
                        print('Running ML algo to confirm the possibility of fire in given frames : .... ')
                        cv2.putText(img=frame, text='Running ML algo to confirm the possibility of fire in given frames : ....' , org=(10, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 240))
                        cv2.imshow("Live Video : ", frame)
                
                        count = 0
                        for i in range(len(Frames_with_Fire)):
                            if (FIF_ML(Frames_with_Fire[i])):
                                count+=1

                        print('Fire detected in '+str(count)+' frames out of '+str(len(Frames_with_Fire)))
                        if len(Frames_with_Fire)-count <= 4 :
                            print('ALERT ! There\'s fire in the kitchen. ')
                            # call the Fire alarm function 
                            Alarms.Fire()

                else :
                    Frames_with_Fire = [] 

            frame_count = 0


        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            break

        

def fire_in_footage():
    Frames_with_Fire = []
    print('You are in option to detect fire in live vedio. ')
    sec = 1
    frame_count = 0
    print('Turning on the camera ... ')
    video = cv2.VideoCapture(0)
    while True:
        print('checking for signs of fire in live vedio input from webcam ..... ')
        # video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        (grabbed, frame) = video.read()
        frame_count+=1
        # will use the algorithm on every 100th frame
        if frame_count>=4:
            cv2.putText(img=frame, text='checking if there\'s someone in the room ............' , org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            cv2.imshow("Live Video : ", frame)

            # print('checking if there\'s someone in the room ............')
            persons_in_room = person_detector(frame)
            # persons_in_room = person_detector_temp(frame)
            cv2.putText(img=frame, text='Number of persons in room : '+str(len(persons_in_room)), org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 0))
            cv2.imshow("Live Video : ", frame)

            if (len(persons_in_room)==0):
                print('checking for signs of fire in frame .............. ')
                cv2.putText(img=frame, text='There\'s no one in the room so checking for signs of fire in frame .............. ', org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 0))
                cv2.imshow("Live Video : ", frame)
                
                if (FIF_IP(frame)):
                    Frames_with_Fire.append(frame)
                    print('Detected fire in '+str(len(Frames_with_Fire))+' Consequent Frames. ')
                    cv2.putText(img=frame, text='Detected fire in '+str(len(Frames_with_Fire))+' Consequent Frames. ', org=(10, 55), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 0))
                    cv2.imshow("Live Video : ", frame)
                
                    if (len(Frames_with_Fire) >= 10 ):
                        print('Running ML algo to confirm the possibility of fire in given frames : .... ')
                        cv2.putText(img=frame, text='Running ML algo to confirm the possibility of fire in given frames : ....' , org=(10, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 240))
                        cv2.imshow("Live Video : ", frame)
                
                        count = 0
                        for i in range(len(Frames_with_Fire)):
                            if (FIF_ML(Frames_with_Fire[i])):
                                count+=1

                        print('Fire detected in '+str(count)+' frames out of '+str(len(Frames_with_Fire)))
                        if len(Frames_with_Fire)-count <= 4 :
                            print('ALERT ! There\'s fire in the kitchen. ')
                            # call the Fire alarm function 
                            Alarms.Fire()

                else :
                    Frames_with_Fire = [] 

            frame_count = 0 

        if not grabbed:
            print('not grabbed')
            video.release()
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video.release()
            break

# deploy GUI
if __name__=='__main__':

    root = tk.Tk()
    root.title('Fire Detection App')
    root.geometry("720x580")
    # Create 2 frames for interface
    leftFrame = Frame(root)
    label2 = Label(leftFrame, text="Fire Detection System. ",fg="Red",  font=('Verdana', 25, 'bold'))
    label2.pack()
    quote = """
    We are trying to keep your loved ones safe from hazards \nthat are frequent reasons of injuries and loss among\n elederly people. 
    \n You can do other works while leaving your\n loved ones in our system\'s safe hands. """
    label0 = Label(leftFrame, text=quote,fg="Green", font=('Verdana', 15, 'bold'))
    label0.pack()
    label1 = Label(leftFrame, text="\n\nSelect one of the following options : \n Check fire in a : \n ", font=('Verdana', 15, 'bold'))
    label1.pack()
    buttonSeek = Button(leftFrame, text="Picture", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 22, 'bold', 'italic'), command=lambda : fire_in_image())
    buttonSeek.pack(side=LEFT)

    buttonSeek = Button(leftFrame, text="Video", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('BioRhyme', 22, 'bold', 'italic'), command=lambda : fire_in_video())
    buttonSeek.pack(side=LEFT)

    leftFrame.pack(padx=40, pady=10)    

    rightFrame = Frame(root)
    buttonSeek = Button(rightFrame, text="Live Video", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 22, 'bold', 'italic'), command=lambda : fire_in_footage())
    buttonSeek.pack()
    rightFrame.pack()

    root.mainloop()



