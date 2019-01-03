from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from _thread import start_new_thread as st

import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import time
import threading

from Source.packages import facenet, detect_face

modeldir = '../Model'
list=os.listdir('../Model/class')
print(list)
classifier_filename = '../Model/class/'+str(list[-1])
print(classifier_filename)
s=classifier_filename.split('/')
s2=s[-1].split('_')
s1=str(s2[1])+":"+str(s2[2])
print(s1)
npy = ''
train_img = "Train_data"
size = 640,480, 3
video_frame = np.zeros(size, dtype=np.uint8)
video_capture = cv2.VideoCapture(0)

def read():
    global video_capture
    global video_frame
    while(1):
        ret, frame1 = video_capture.read()
        video_frame=frame1

t1 = threading.Thread(target=read)
t1.start()
flag = 0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 100
        image_size = 182
        input_image_size = 160

        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

name=[]

def Register(frame):
    global flag
    c=0
    curTime = time.time() + 1  # calc fps
    timeF = frame_interval
    if (c % timeF == 0):
        find_results = []

        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Detected_FaceNum: %d' % nrof_faces)

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(frame.shape)[0:2]

            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    print('Face is very close!')
                    continue
                img1 = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]+50]
                Width = bb[i][2] - bb[i][0]
                Height = bb[i][3] - bb[i][1]
                if(Width > 180 and Height > 180):
                    print("Corret dimentions")
                else:
                    print("Provide correct dimensions.........")
                    continue
                #img1 = cv2.resize(img1, (300, 300))

                cv2.imshow("recognised frame", img1)
                cv2.waitKey(10)
                print("Waiting for the key")
                key = input("Enter the y or n")
                if (key == 'Y' or key == 'y'):
                    n1 = input("Enter the name of the person")
                    if (os.path.isdir("Train_data/" + n1) and os.path.isdir("pre_img/" + n1)):
                        print("dir exists")
                    else:
                        os.mkdir("Train_data/" + n1)
                        os.mkdir("pre_img/" + n1)
                    cv2.imwrite("Train_data/" + n1 + "/image" + ".jpg", frame)
                    fra=frame
                    key1 = 25
                    for m in range(key1):
                        cv2.imwrite("pre_img/" + n1 + "/image" + str(m) + ".jpg", img1)
                    cv2.putText(frame, str(n1), (100, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),
                                thickness=1,lineType=2)
                    cv2.imshow("name",fra)
                    cv2.waitKey(10)
                    from Source.classifier_train import classy
                    classy()
                    flag = 1
    return name


cnt=0
cnt1=0

#video_capture = cv2.VideoCapture(0)
#video_capture = cv2.VideoCapture("rtsp://admin:innominds123$@192.168.202.250:554")
print("R:Register" + "\n" + "L:List of classes" + "\n" + "T:TimeStamp" + "\n" + "M:Model")
while(1):
        list = os.listdir('../Model/class')
        classifier_filename = '../Model/class/' + str(list[-1])
        s = classifier_filename.split('/')
        s2 = s[-1].split('_')
        s1 = str(s2[1]) + ":" + str(s2[2])
        HumanNames = os.listdir(train_img)
        for i in HumanNames:
            if ("Unknown" not in i and "Unknown-1" not in i):
                name.append(i)
        frame1 = video_frame.copy()
        a=np.array(frame1)
        if(None not in a.ravel()):
            frame = frame1[80:460, 100:590]
            cv2.rectangle(frame1, (100, 80), (590, 460), (0, 255, 255), 4)
            data = ["R:Register", "L:Classes list", "T:Timestamp", "M:Model"]
            y = 25
            for i in data:
                cv2.putText(frame1, str(i), (15, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1,
                            lineType=2)

                y += 25
            x = 25
            for i, j in enumerate(name):
                #print("name:",name)
                final = "Id" + str(i) + ":" + str(j)
                cv2.putText(frame1, str(final), (450, x), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1,
                            lineType=2)
                x += 25
            if cv2.waitKey(10) & 0xFF == ord('r'):
                st(Register, (frame,))
                #Register(frame, HumanNames)
            elif cv2.waitKey(10) & 0xFF == ord('l'):
                print("Total no. of classes: ", len(HumanNames))
                print("List of classes: ", HumanNames)
                cv2.putText(frame1, str(len(HumanNames)), (200, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0),
                            thickness=1,
                            lineType=2)

            elif cv2.waitKey(10) & 0xFF == ord('t'):
                t = time.localtime()
                t1 = time.asctime(t)
                print("current time: %s " % t1)
                cv2.putText(frame1, str(t1), (200, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1,
                            lineType=2)

            elif cv2.waitKey(10) & 0xFF == ord('m'):
                print("Model_name:", s[-1])
                cv2.putText(frame1, str(s[-1]), (200, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1,
                            lineType=2)

            cv2.putText(frame1, str(s[-1]), (150, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), thickness=1,
                        lineType=2)
            cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame1)
            name=[]
video_capture.release()
cv2.destroyAllWindows()














