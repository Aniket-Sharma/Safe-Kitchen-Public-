# Imports the Google Cloud client library and streaming libraries
import io
import os
import cv2

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

def person_detector_temp(image):
    return []

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

    cv2.imshow("Live Video", frame)
    cv2.imshow("Probabilty of fire :", output)

    #print("output:", frame)
    #print(int(no_red))
    
    if int(no_red) > 20000:
        print ('Fire detected in this image. ')
        return True
    print('There are '+str(prob_of_fire)+' %  chances of fire in this frame. ')
    return False


def FIF_ML(frame):
    # retrun True or False based on whether the ML algo detects fire in image or not. 
    return True


def getFrame(vidcap, sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

def Vedio_to_frames(video):
    vidcap = cv2.VideoCapture('video.mp4')
    sec = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(vidcap, sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    
# Consumes google.api.vision
def func():
    # The name of the image file to annotate
    file_path = filedialog.askopenfilename()
    # set enviromental variable to the API credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cred.json"
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    bandera = False    
    
    if bandera == False:
        messagebox.showinfo("Safe Area", "It all looks alright !")
    else:
        danger_msg = str(actor)+" could be in danger from "+str(danger)
        messagebox.showinfo("Danger", danger_msg)


def fire_in_frame(model, label_dict, frame):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cred.json"
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_for_pred = prepare_image_for_prediction( resized_frame )
    pred_vec = model.predict(frame_for_pred)
    #print(pred_vec)
    pred_class =[]
    confidence = np.round(pred_vec.max(),2)
    if confidence > 0.4:
        pc = pred_vec.argmax()
        pred_class.append( (pc, confidence) )
    else:
        pred_class.append( (0, 0) )
    if pred_class:
        txt = get_display_string(pred_class, label_dict)
        frame = draw_prediction( frame, txt )
    #print(pred_class)
    #plt.axis('off')
    #plt.imshow(frame)
    #plt.show()
    #clear_output(wait = True)
def fire_in_image():
	print('You are in option to detect fire in an image.')
	file_path = filedialog.askopenfilename()
	with io.open(file_path, 'rb') as image_file:
		content = image_file.read()
	
	return
    
def fire_in_video():
	print('You are in option to detect fire in a video.')
	video_path = filedialog.askopenfilename()
	with io.open(video_path, 'rb') as video_file:
            vs = cv2.VideoCapture(video_file)
	fps = math.floor(vs.get(cv2.CAP_PROP_FPS))
	ret_val = True
	writer = 0
	while True:
            ret_val, frame = vs.read()
            if not ret_val:
                break
            resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
		# frame_for_pred = prepare_image_for_prediction( resized_frame )
		# pred_vec = model.predict(frame_for_pred)
		# pred_class =[]
		# confidence = np.round(pred_vec.max(),2)
		# if confidence > 0.4 :
		# 	pc = pred_ves.argmax()
		# 	pred_class.append( (pc, confidence) )
		# else:
		# 	pred_class.append( (0, 0) )
		# if pred_class:
		# 	txt = get_display_string(pred_class, label_dict)
		# 	frame = draw_prediction( frame, txt )
            persons_in_room = person_detector(frame)
            if (len(persons_in_room)==0):
                print('checking for signs of fire in frame .............. ')
                if (FIF_IP(frame)):
                    Frames_with_Fire.append(frame)
                    print('Detected fire in '+str(len(Frames_with_Fire))+' Consequent Frames. ')
                    if (len(Frames_with_Fire) >= 10 ):
                        print('Running ML algo to confirm the possibility of fire in given frames : .... ')
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

Frames_with_Fire = []
def fire_in_footage():
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
            # print('checking if there\'s someone in the room ............')
            persons_in_room = person_detector(frame)
            # persons_in_room = person_detector_temp(frame)
            if (len(persons_in_room)==0):
                print('checking for signs of fire in frame .............. ')
                if (FIF_IP(frame)):
                    Frames_with_Fire.append(frame)
                    print('Detected fire in '+str(len(Frames_with_Fire))+' Consequent Frames. ')
                    if (len(Frames_with_Fire) >= 10 ):
                        print('Running ML algo to confirm the possibility of fire in given frames : .... ')
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
	root.geometry("520x480")
	# Create 2 frames for interface
	leftFrame = Frame(root)
	leftFrame.pack(side=LEFT, padx=25, pady=20)
	#Create widgets for 1st frame
	label1 = Label(leftFrame,text="Keeping you safe from fire")
	label1.pack(side=TOP)
	label2 = Label(leftFrame,text="Select one of the options")
	label2.pack(side=TOP)
	buttonSeek = Button(leftFrame, text="Check fire in a picture", fg="Blue", command=lambda : fire_in_image())
	buttonSeek.pack(side=TOP)
	buttonSeek = Button(leftFrame, text="Check fire in a video", fg="Yellow", command=lambda : fire_in_video())
	buttonSeek.pack(side=TOP)
	buttonSeek = Button(leftFrame, text="Check fire in live video from your webcam", fg="Red", command=lambda : fire_in_footage())
	buttonSeek.pack(side=TOP)

	root.mainloop()




