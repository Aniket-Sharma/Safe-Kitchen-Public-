# Imports the Google Cloud client library and streaming libraries
import io
import os
import cv2
import math
import time 

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

fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0

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

#detecting fall 
def fall_detector(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    
    #Find contours
    contours,_  = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        areas = []

        for contour in contours: 	
            ar = cv2.contourArea(contour)
            areas.append(ar)
        
        max_area = max(areas or [0])

        max_area_index = areas.index(max_area)

        cnt = contours[max_area_index]

        M = cv2.moments(cnt)
        
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
        
        if h < w:
            # j += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow('Fall detected', frame)
            cv2.waitKey(150)
            return True
            
        # if j > 10:
        #     #print "FALL"
        #     #cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255,255,255), 2)
        #     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        #     Alarms.Fall()

        if h > w:
            # j = 0 
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow('person seems to be alright. ', frame)
            cv2.waitKey(50)
            return False
        
#detecting the persons present in the room.
        
def person_detector(image):
    print('checking if there\'s someone in the room ............')
    # retruns the reactangle array highlithing the location of person in given image.  
    orig = image.copy()
    (rects, weights) = HOGCV.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    result = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    if len(result)==0 :
        print('There\'s no person in the image. ')
    else:
        print('There are '+str(len(result))+' persons in the image.')
    return result

#marking the persons present in the room. 
def mark_person(image, pick): 
    print('we are here. ')
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (255,255,0), 2)
    cv2.imshow("Persons in Image : ", image)
    return 
    
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
    cv2.imshow("Pixels with Fire like properties : " , output)
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
    content = cv2.imread(file_path)
    image = content.copy()
    content = imutils.resize(content, width=min(400, content.shape[1]))
    cv2.imshow("Original Image : ", content)
    
    persons_in_room = person_detector(content)
    # persons_in_room = person_detector_temp(frame)

    mark_person(content, persons_in_room)
    cv2.waitKey(600)

    leftFrame = tk.Tk()

    label4 = Label(leftFrame, text="Number of Persons in the Image : "+str(len(persons_in_room)), fg="Green", font=('Verdana', 15, 'bold'))
    label4.pack()

    label7 = Label(leftFrame, text="Checking if the person has fallen or not . ... ", fg="Green", font=('Verdana', 15, 'bold'))
    label7.pack()

    if (fall_detector(image)):
    	label8 = Label(leftFrame, text="The person might have fallen ... ", fg="Green", font=('Verdana', 15, 'bold'))
    	label8.pack()
        
    else :
    	label7 = Label(leftFrame, text="Person seems to be doing alright. ", fg="Green", font=('Verdana', 15, 'bold'))
    	label7.pack()

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
    Frames_with_Fire = []
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

        if frame_count>=1:

            cv2.putText(img=frame, text='checking if there\'s someone in the room ............' , org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            cv2.imshow("Live Video : ", frame)

            # print('checking if there\'s someone in the room ............')
            persons_in_room = person_detector(frame)
            # persons_in_room = person_detector_temp(frame)
            cv2.putText(img=frame, text='Number of persons in room : '+str(len(persons_in_room)), org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 0))

            cv2.imshow("Live Video : ", frame)
            mark_person(frame, persons_in_room)
            
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

            else :
            	print('Detecting fall .... ')
            	frames_with_fall = 0 
            	if (fall_detector(frame)):
            		frames_with_fall+=1
            		print('The person seems to be fallen, please stand up quickly to avoid the alarm. ')
            		if frames_with_fall > 5:
            			print('PLease be patient, your emergency contacts will be notified. ')
            			Alarms.Fall()
            		else :
            		    frames_with_fall = 0 


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
        if frame_count>=2:
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

            else:
            	print('Detecting fall .... ')
            	frames_with_fall = 0 
            	if (fall_detector(frame)):
            		frames_with_fall+=1
            		print('The person seems to be fallen, please stand up quickly to avoid the alarm. ')
            		if frames_with_fall > 5:
            			print('PLease be patient, your emergency contacts will be notified. ')
            			Alarms.Fall()
            		else :
            		    frames_with_fall = 0  

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
    root.title('Fire and Fall Detection App')
    root.geometry("720x580")
    # Create 2 frames for interface
    leftFrame = Frame(root)
    label2 = Label(leftFrame, text="Fire and Fall Detection System. ",fg="Red",  font=('Verdana', 25, 'bold'))
    label2.pack()
    quote = """
    We are trying to keep your loved ones safe from hazards \nthat are frequent reasons of injuries and loss among\n elederly people. 
    \n You can do other works while leaving your\n loved ones in our system\'s safe hands. """
    label0 = Label(leftFrame, text=quote,fg="Green", font=('Verdana', 15, 'bold'))
    label0.pack()
    label1 = Label(leftFrame, text="\n\nSelect one of the following options : \n Check fire/fall in a : \n ", font=('Verdana', 15, 'bold'))
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



