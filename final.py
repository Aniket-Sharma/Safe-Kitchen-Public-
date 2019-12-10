
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)'
import matplotlib.pyplot as plt
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.python.keras import optimizers
from tkinter import filedialog

import cv2
import math
from IPython.display import clear_output

import os

#IP and CV

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

fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0

# Opencv pre-trained SVM with HOG people features 

# HOGCV = cv2.HOGDescriptor()
# HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


#ML
print(os.listdir("input/fire-detection-from-cctv/data/img_data"))
print(os.listdir("input/fire-detection-from-cctv/data/video_data/test_videos"))
print(os.listdir("input/fire-detection-from-cctv/data"))

print(os.listdir("input/resnet50"))
print(os.listdir("input/vgg16"))

#global constansts in case of complete training

IMG_SIZE = 20
NUM_EPOCHS = 1
NUM_CLASSES = 3
TRAIN_BATCH_SIZE = 77
TEST_BATCH_SIZE = 1 

# constants for fast training

FAST_IMG_SIZE = 20
FAST_NUM_EPOCHS = 1
#NUM_CLASSES = 3
FAST_TRAIN_BATCH_SIZE = 77
FAST_TEST_BATCH_SIZE = 1 

#trained_model_l, label_dict_l, model, train_generator, validation_generator, trained_model_s, label_dict_s = {}, {}, {}, {}, {}, {}, {}

def create_model( model_size ):
    my_new_model = Sequential()
    if  model_size == 'L':
        resnet_weights_path = 'input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        resnet = ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path)
        #resnet.summary()
        my_new_model.add(resnet)
        my_new_model.layers[0].trainable = False
    else:
        vgg_weights_path = 'input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        vgg= VGG16(include_top=False, weights=vgg_weights_path ) 
        vgg.summary()
        my_new_model.add(vgg)
        my_new_model.add(GlobalAveragePooling2D())
        my_new_model.layers[0].trainable = False
        my_new_model.layers[1].trainable = False
        
    my_new_model.add(Dense(NUM_CLASSES, activation='softmax'))
   
    # Say no to train first layer (ResNet) model. It is already trained
    
    opt = optimizers.adam()
    my_new_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return my_new_model

# %% [markdown]
# ## Training
# 
# The frames for training and testing are read from the directories using ImageDataGenerator.Data augmentation is performed by horizontally flipping each image by setting the horizontal_flip to True in the ImageDataGenerator.

# %% [code]
def train_model( model, way='fast'):
    #ata_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    if way=='fast':
        data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                #sear_range=0.01,
                                zoom_range=[0.9, 1.25],
                                horizontal_flip=True,
                                vertical_flip=False,
                                data_format='channels_last',
                                brightness_range=[0.5, 1.5]
                               )
        train_generator = data_generator_with_aug.flow_from_directory('input/fire-detection-from-cctv/data/data/img_data/train', target_size=(FAST_IMG_SIZE, FAST_IMG_SIZE), batch_size=FAST_TRAIN_BATCH_SIZE, class_mode='categorical')
        validation_generator = data_generator_with_aug.flow_from_directory('input/fire-detection-from-cctv/data/data/img_data/test', target_size=(FAST_IMG_SIZE, FAST_IMG_SIZE), batch_size=FAST_TEST_BATCH_SIZE, shuffle = False, class_mode='categorical')
        H = model.fit_generator(train_generator, steps_per_epoch=train_generator.n/TRAIN_BATCH_SIZE, epochs=FAST_NUM_EPOCHS, validation_data=validation_generator, validation_steps=1)
        return model, train_generator,validation_generator
    
    else:
        data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                #sear_range=0.01,
                                zoom_range=[0.9, 1.25],
                                horizontal_flip=True,
                                vertical_flip=False,
                                data_format='channels_last',
                                brightness_range=[0.5, 1.5]
                               )
        train_generator = data_generator_with_aug.flow_from_directory(
            'input/fire-detection-from-cctv/data/data/img_data/train',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=TRAIN_BATCH_SIZE,
            class_mode='categorical')

        validation_generator = data_generator_with_aug.flow_from_directory(
            'input/fire-detection-from-cctv/data/data/img_data/test',
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=TEST_BATCH_SIZE,
            shuffle = False,
            class_mode='categorical')
    
        
    #y_train = get_labels(train_generator)
    #weights = class_weight.compute_class_weight('balanced',np.unique(y_train), y_train)
    #dict_weights = { i: weights[i] for i in range(len(weights)) }

        H = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.n/TRAIN_BATCH_SIZE,
            epochs=NUM_EPOCHS,
            validation_data=validation_generator,
            validation_steps=1 #,
            #class_weight=dict_weights
                )
    
    #plot_history( H, NUM_EPOCHS )
        return model, train_generator,validation_generator

# %% [code]
def get_label_dict(train_generator ):
# Get label to class_id mapping
    labels = (train_generator.class_indices)
    label_dict = dict((v,k) for k,v in labels.items())
    return  label_dict   

# %% [code]
def get_labels( generator ):
    generator.reset()
    labels = []
    for i in range(len(generator)):
        labels.extend(np.array(generator[i][1]) )
    return np.argmax(labels, axis =1)

# %% [code]
def get_pred_labels( test_generator):
    test_generator.reset()
    pred_vec=model.predict_generator(test_generator,
                                     steps=test_generator.n, #test_generator.batch_size
                                     verbose=1)
    return np.argmax( pred_vec, axis = 1), np.max(pred_vec, axis = 1)
    

def draw_prediction( frame, class_string ):
    x_start = int(frame.shape[0]/2)
    y_start = int(frame.shape[1]/2) 
    cv2.putText(frame, class_string, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
    return frame

def prepare_image_for_prediction( img):
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def get_display_string(pred_class, label_dict):
    txt = ""
    for c, confidence in pred_class:
        txt += label_dict[c]
        if c :
            txt += '['+ str(confidence) +']'
    return txt

#train the model by default 

model = create_model('L')
trained_model_l, train_generator,validation_generator = train_model(model, way='fast')
label_dict_l = get_label_dict(train_generator )

model = create_model('S')

trained_model_s, train_generator,validation_generator = train_model(model, way='fast')
label_dict_s = get_label_dict(train_generator)


def ML_main():

	global trained_model_l, label_dict_l, model, train_generator, validation_generator, trained_model_s, label_dict_s

	model = create_model('L')
	trained_model_l, train_generator,validation_generator = train_model(model, way='slow')
	label_dict_l = get_label_dict(train_generator )

	model = create_model('S')

	trained_model_s, train_generator,validation_generator = train_model(model, way='slow')
	label_dict_s = get_label_dict(train_generator)


#returns True or False based on possibility of fire or smoke. 
def FIF_ML(frame, model=trained_model_l, label_dict=label_dict_l ):       
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame_for_pred = prepare_image_for_prediction( resized_frame )
    pred_vec = model.predict(frame_for_pred)
    print(pred_vec)
    pred_class =[]
    confidence = np.round(pred_vec.max(),2)

    fire_in_frame = False

    if confidence > 0.4:
        pc = pred_vec.argmax()
        pred_class.append( (pc, confidence) )
        fire_in_frame = True

    else:
        pred_class.append( (0, 0) )

    if pred_class:
        txt = get_display_string(pred_class, label_dict)       
        frame = draw_prediction( frame, txt )

    print("This frame\'s pred class is : "+str(pred_class))
    print("The confidence rate in this frame is : "+str(confidence))

    plt.axis('off')
    plt.imshow(frame)
    plt.show()
    clear_output(wait = True)

    return fire_in_frame 


	# while True:
	# 	video_path = filedialog.askopenfilename()
	# 	predict ( trained_model_l, video_path, 'test1_9.avi',  label_dict_l)
		
	# 	if 0xFF == ord('q'):
	# 		break


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

#detecting smoke in a frame using image processing
def SIF_IP(frame):
    a1, a2, k1, k2, k3, k4 = 5, 20, 80, 150, 190, 255 
    height, width, _ = frame.shape
    smoke = 0 
    for i in range(height):
        for j in range(width):
            m = max(frame[i][j][0], frame[i][j][1], frame[i][j][2])
            n = min(frame[i][j][0], frame[i][j][1], frame[i][j][2])
            intensity = (frame[i][j][0]+frame[i][j][1]+frame[i][j][2]) / 3
            a = m-n
            if a <= a1 and (intensity>=k1 and intensity<=k2):
                frame[i][j] = [0, 255, 0]
                smoke+=1
            elif a <= a2 and (intensity>=k3 and intensity<=k4):
                frame[i][j] = [0,0,255]
                smoke+=1

    prob = smoke*100 / width*height
    
    print('Smoke Pixel percentage  : '+str(prob)+' % ')
    
    if prob>10:
        cv2.imshow("Smoke Pixels heighlited  : " , frame)
        cv2.waitKey(500)
        return True
    return False
            
    
# detecting fire in a frame using image processing 
def FIF_IP(frame):
    cv2.imshow("Original" , frame)
    cv2.waitKey(10)
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

    width, height, _ = frame.shape
    
    prob_of_fire = (int(no_red)*100)/(width*height)
    
    print('Chances of fire : '+str(prob_of_fire)+' % ')
    cv2.putText(img=output, text='Chances of fire : '+str(prob_of_fire)+' % ' , org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 0))
    
    # cv2.imshow("Live Video : ", frame)

    # cv2.imshow("Probabilty of fire : ", output)
    cv2.imshow("Pixels with Fire like properties : " , output)
    cv2.waitKey(10)
    # cv2.putText(img, prob_of_fire, (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
    #print("output:", frame)
    #print(int(no_red))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return False

    if int(prob_of_fire) > 20:
        print ('Fire detected in this image. ')
        return True

    print('There are '+str(prob_of_fire)+' %  chances of fire in this frame. ')

    return False


def fire_in_image():

    print('You are in option to detect fire in an image.')
    file_path = filedialog.askopenfilename()
    image = content = cv2.imread(file_path)
    content = imutils.resize(content, width=min(400, content.shape[1]))
    cv2.imshow("Original Image : ", content)
    # content = np.float32(content)
    # content.astype(np.float32).dtype
    #root = tk.Tk()
    #root.title('Fire Detection App')
    #root.geometry("720x580")
    #leftFrame = Frame(root)
    #label2 = Label(leftFrame, text="Detecting fire in an Image. ",fg="Red",  font=('Verdana', 25, 'bold'))
    #label2.pack()

    #label0 = Label(leftFrame, text="Please wait while the system looks for signs of fire in given image. ", fg="Green", font=('Verdana', 15, 'bold'))
    #label0.pack()
    
    #cv2.putText(img=content, text='checking if there\'s someone in the room ............' , org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
    #cv2.imshow("Original Image : ", content)

    persons_in_room = person_detector(content)
    # persons_in_room = person_detector_temp(frame)
    mark_person(content, persons_in_room)
    #cv2.putText(img=content, text='Number of persons in room : '+str(len(persons_in_room)), org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 0))
    #cv2.imshow("Live Video : ", content)
    cv2.waitKey(600)
    leftFrame = tk.Tk()

    content = cv2.imread(file_path)
    content = imutils.resize(content, width=min(400, content.shape[1]))
    cv2.imshow("Original Image : ", content)
    
    persons_in_room = person_detector(content)
    # persons_in_room = person_detector_temp(frame)
    mark_person(content, persons_in_room)
    cv2.waitKey(600)

    leftFrame = tk.Tk()

    label4 = Label(leftFrame, text="Number of Persons in the Image : "+str(len(persons_in_room)), fg="Green", font=('Verdana', 15, 'bold'))
    label4.pack()

    if (FIF_IP(content) or SIP_IP(content)):

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
            mark_person(frame, persons_in_room)
            
            if (len(persons_in_room)==0):
                print('checking for signs of fire in frame .............. ')
                cv2.putText(img=frame, text='There\'s no one in the room so checking for signs of fire in frame .............. ', org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 0))
                cv2.imshow("Live Video : ", frame)
                
                if (FIF_IP(frame) or SIP_IP(frame)):
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
            new_frame = imutils.resize(frame, width=min(400, frame.shape[1]))
            # print('checking if there\'s someone in the room ............')
            persons_in_room = person_detector(new_frame)
            if (len(persons_in_room)>=1):
                print('The fall detection algorithm will work now.. ')
                mark_person(new_frame, persons_in_room)
                cv2.waitKey(50)
                fall_detection(frame)
            # persons_in_room = person_detector_temp(frame)
            cv2.putText(img=frame, text='Number of persons in room : '+str(len(persons_in_room)), org=(10, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(0, 0, 0))
            cv2.imshow("Live Video : ", frame)

            if (len(persons_in_room)==0):
                print('checking for signs of fire in frame .............. ')
                cv2.putText(img=frame, text='There\'s no one in the room so checking for signs of fire in frame .............. ', org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5, color=(255, 255, 0))
                cv2.imshow("Live Video : ", frame)
                
                if (FIF_IP(frame) or SIP_IP(frame)):
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
#Image Processing Functions
def ip_person_in_image():
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)
    cv2.imshow("Original Image : ", image)
    cv2.waitKey(0)
    person_detector(image)
    return 

def ip_fire_in_image():
    file_path = filedialog.askopenfilename()
    image = content = cv2.imread(file_path)
    content = imutils.resize(content, width=min(400, content.shape[1]))
    cv2.imshow("Original Image : ", content)
    cv2.waitKey(0)
    FIF_IP(content)
    return

def ip_fall_in_image():
    file_path = filedialog.askopenfilename()
    image = cv2.imread(file_path)
    #content = imutils.resize(content, width=min(400, content.shape[1]))
    cv2.imshow("Original Image : ", image)
    fall_detector(image)
    return

def ip_fall_in_video():
    return

def ip_fire_in_video():
    return
def ip_person_in_video():
    return

def ip_fall_in_footage():
    return 
def ip_fire_in_footage():
    return
def ip_preson_in_footage():
    return

# Machine Learning Functions
def ml_fire_in_image():
    return
def ml_fire_in_video():
    return
def ml_fire_in_footage():
    return


def ML_functions():

    root = tk.Tk()
    root.title('Fire and Fall Detection App')
    root.geometry("720x580")
    # Create 2 frames for interface
    leftFrame = Frame(root)
    label2 = Label(leftFrame, text="Machine Learning Fire Detection System  ",fg="Red",  font=('Verdana', 25, 'bold'))
    label2.pack()
    # quote = """
    # We are trying to keep your loved ones safe from hazards \nthat are frequent reasons of injuries and loss among\n elderly people. 
    # \n You can do other works while leaving your\n loved ones in our system\'s safe hands. """
    # label0 = Label(leftFrame, text=quote,fg="Green", font=('Verdana', 15, 'bold'))
    # label0.pack()
    label1 = Label(leftFrame, text="\n\nSelect one of the following options : \n Check fire in a : \n ", font=('Verdana', 15, 'bold'))
    label1.pack()
    buttonSeek = Button(leftFrame, text="Picture", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 22, 'bold', 'italic'), command=lambda : ml_fire_in_image())
    buttonSeek.pack(side=LEFT)

    buttonSeek = Button(leftFrame, text="Video", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('BioRhyme', 22, 'bold', 'italic'), command=lambda : ml_fire_in_video())
    buttonSeek.pack(side=LEFT)

    leftFrame.pack(padx=40, pady=10)    

    rightFrame = Frame(root)
    buttonSeek = Button(rightFrame, text="Live Video", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 22, 'bold', 'italic'), command=lambda : ml_fire_in_footage())
    buttonSeek.pack()

    rightFrame.pack()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.destroy()
        return

    if 0xFF == ord('e'):
        root.destroy()
        return

    root.mainloop()


def IP_functions():
    root = tk.Tk()
    root.title('Fire and Fall Detection App')
    root.geometry("720x580")
    # Create 2 frames for interface
    leftFrame = Frame(root)
    label2 = Label(leftFrame, text="Image Processing Fire and Fall Detection System. ",fg="Red",  font=('Verdana', 25, 'bold'))
    label2.pack()
    quote = """
    We are trying to keep your loved ones safe from hazards \nthat are frequent reasons of injuries and loss among\n elderly people. 
    \n You can do other works while leaving your\n loved ones in our system\'s safe hands. """
    label0 = Label(leftFrame, text=quote,fg="Green", font=('Verdana', 15, 'bold'))
    label0.pack()
    label1 = Label(leftFrame, text="\n\nSelect one of the following options : \n ", font=('Verdana', 15, 'bold'))
    label1.pack()
    buttonSeek = Button(leftFrame, text="Fire detection in Picture", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 15, 'bold', 'italic'), command=lambda : ip_fire_in_image())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(leftFrame, text="Fall detection in Picture", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 15, 'bold', 'italic'), command=lambda : ip_fall_in_image())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(leftFrame, text="Person detection in Picture", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 15, 'bold', 'italic'), command=lambda : ip_person_in_image())
    buttonSeek.pack(side=LEFT)
    leftFrame.pack(padx=10, pady=10)

    centerFrame = Frame(root)
    buttonSeek = Button(centerFrame, text="Fire detection in Video", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('BioRhyme', 15, 'bold', 'italic'), command=lambda : ip_fire_in_video())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(centerFrame, text="Fall detection in Video", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 15, 'bold', 'italic'), command=lambda : ip_fall_in_video())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(centerFrame, text="Person detection in Video", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 15, 'bold', 'italic'), command=lambda : ip_person_in_video())
    buttonSeek.pack(side=LEFT)

    centerFrame.pack(padx=10, pady=10)    

    rightFrame = Frame(root)
    buttonSeek = Button(rightFrame, text="Fire detection in Live Video", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 15, 'bold', 'italic'), command=lambda : ip_fire_in_footage())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(rightFrame, text="Fall detection in Live Video", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 15, 'bold', 'italic'), command=lambda : ip_fall_in_footage())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(rightFrame, text="Person detection in Live Video", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 15, 'bold', 'italic'), command=lambda : ip_person_in_footage())
    buttonSeek.pack(side=LEFT)
    rightFrame.pack(padx=10, pady=10)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        root.destroy()
        return

    if 0xFF == ord('e'):
        root.destroy()
        return

    root.mainloop()


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
    We are trying to keep your loved ones safe from hazards \nthat are frequent reasons of injuries and loss among\n elderly people. 
    \n You can do other works while leaving your\n loved ones in our system\'s safe hands. """
    label0 = Label(leftFrame, text=quote,fg="Green", font=('Verdana', 15, 'bold'))
    label0.pack()
    label1 = Label(leftFrame, text="\n\nSelect one of the following options : \n Check fire, smoke or fall in a : \n ", font=('Verdana', 15, 'bold'))
    label1.pack()
    buttonSeek = Button(leftFrame, text="Picture", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('Rakkas', 22, 'bold', 'italic'), command=lambda : fire_in_image())
    buttonSeek.pack(side=LEFT)

    buttonSeek = Button(leftFrame, text="Video", fg="Blue", bg="#DBF7DD",width=20, padx=20, pady=10 , font=('BioRhyme', 22, 'bold', 'italic'), command=lambda : fire_in_video())
    buttonSeek.pack(side=LEFT)

    leftFrame.pack(padx=40, pady=10)    

    rightFrame = Frame(root)
    buttonSeek = Button(rightFrame, text="Live Video", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 22, 'bold', 'italic'), command=lambda : fire_in_footage())
    buttonSeek.pack(side=LEFT)
    rightFrame.pack(padx=40, pady=10)

    otherframe = Frame(root)
    buttonSeek = Button(otherframe, text="Train the ML Model ", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 12, 'bold', 'italic'), command=lambda : ML_main())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(otherframe, text="Image Processing Functions", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 12, 'bold', 'italic'), command=lambda : IP_functions())
    buttonSeek.pack(side=LEFT)
    buttonSeek = Button(otherframe, text="Machine Learning Functions", bg="#DBF7DD", fg="Red" , width=20, padx=20, pady=10 , font=('Classic', 12, 'bold', 'italic'), command=lambda : ML_functions())
    buttonSeek.pack(side=LEFT)
    otherframe.pack(padx=40, pady=40)

    root.mainloop()
        
