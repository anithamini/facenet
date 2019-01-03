from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pickle
import time

import cv2
import numpy as np
import tensorflow as tf
import threading
from scipy import misc

from Source.packages import facenet, detect_face

modeldir = '../Model'
list=os.listdir('../Model/class')
classifier_filename = '../Model/class/'+str(list[-1])
print("LIST:",classifier_filename)
t=classifier_filename.split('/')
s2=t[-1].split('_')
x=s2[3].split(".")
s1="TIME:"+str(s2[1])+":"+str(s2[2])+":"+str(x[0])
npy = ''
train_img = "Train_data"
cnt=0
cnt1=0
size =640,480,3
#video_capture = cv2.VideoCapture("rtsp://admin:innominds123$@192.168.202.250:554")
video_capture = cv2.VideoCapture(0)
video_frame = np.zeros(size,dtype = np.uint8)

#def update_frame():
#    while(1):
#        print("thread...................")
#t = threading.Thread(target=update_frame)
#t.start()
def read():
    global video_capture
    global video_frame
    while (1):
        ret, frame1 = video_capture.read()
        video_frame = frame1

t1 = threading.Thread(target=read)
t1.start()

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
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = os.listdir(train_img)
        HumanNames.sort()

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
        names = []
        for i in class_names:
            if ("bounding boxes" not in i):
                names.append(i)
        for i in class_names:
            if("bounding boxes" not in i and "Unknown" not in i and "Unkonwn-1 not in i"):
               name.append(i)

        print("Opening the camera")

        c = 0

        print('Start Recognition')
        prevTime = 0
        buf=len(list)
        while True:
            list = os.listdir('//IM-RT-IT-158/Users/akesiboyina/Desktop/POC/Model/class')
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            buf1=len(list)
            if(buf1>buf):
                modeldir = '../Model'
                list = os.listdir('../Model/class')
                classifier_filename = '../Model/class/' + str(list[-1])
                print("LIST:", classifier_filename)
                t = classifier_filename.split('/')
                s2 = t[-1].split('_')
                x = s2[3].split(".")
                s1 = "TIME:" + str(s2[1]) + ":" + str(s2[2]) + ":" + str(x[0])
                print('Loading Modal')
                print("**************emb*********************")
                facenet.load_model(modeldir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    print("classnames in emb:**********",class_names)
                name = []
                names=[]
                for i in class_names:
                    if ("bounding boxes" not in i ):
                        names.append(i)
                for i in class_names:
                    if ("bounding boxes" not in i and "Unknown" not in i and "Unkonwn-1 not in i"):
                        name.append(i)
            buf=buf1
            """
            if cnt == 3:
                if cnt1 < 5:
                    cnt1 = cnt1+1
                    ret, frame1 = video_capture.read()
                    continue;
                cnt=0
                cnt1=0
            else:
                ret, frame1 = video_capture.read()
                cnt=cnt+1
            #ret, frame1 = video_capture.read()
            """
            frame1=video_frame.copy()
            a = np.array(frame1)
            if (None not in a.ravel()):

                #frame = frame1[320:700,350:1000]
                #cv2.rectangle(frame1, (350, 250), (1300,1000), (255, 0, 0), 4)
                frame = frame1[80:460, 100:590]
                cv2.rectangle(frame1, (100, 80), (590, 460), (0, 255, 255), 4)
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
                            Width = bb[i][2] - bb[i][0]
                            Height = bb[i][3] - bb[i][1]
                            #if(Width > 100 and Height > 100):
                            #    print("Face detected")
                           # else:
                            #    print("Size of the face is too low")
                             #   break
                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('Face is very close!')
                                break
                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[i] = facenet.flip(cropped[i], False)
                           # cropped[i] =cv2.resize(cropped[i],(180,180))
                            scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            predictions*=100
                            pred = predictions.ravel().tolist()
                            print("pred:",pred)
                            #print("Human names:",HumanNames)
                            print("name:",names)
                            dict1 = dict(zip(names, pred))
                            print(dict1)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]


                            print(best_class_probabilities)

                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0),
                                              2)  # boxing face

                                # plot result idx under box
                            text_x = bb[i][2] - 100
                            text_y = bb[i][1]


                            for H_i in names:

                                if names[best_class_indices[0]] == H_i:
                                    if (best_class_probabilities <45):
                                        Id = "Unknown"
                                        cv2.putText(frame, str(Id), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1, (0, 0, 255), thickness=1, lineType=2)
                                        cv2.putText(frame1,"ACCESS DENIED", (15,25), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 255), thickness=1, lineType=2)
                                        result_names = str(Id)
                                        print("result name:", result_names)
                                        print("\n")


                                    else:
                                        result_names = names[best_class_indices[0]]
                                        text = result_names + ":" + str("%.2f" % (best_class_probabilities))
                                        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                        1.5, (0, 0, 255), thickness=2, lineType=2)
                                        print("result name:", result_names)
                                        print("\n")
                            #cv2.putText(frame1, str(list[-1]), (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                              #  1, (0, 0, 255), thickness=1, lineType=2)

                    else:
                        print('Alignment Failure')

                buf=buf1
                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                x = 25
                for i, j in enumerate(name):
                    final = "Id" + str(i) + ":" + str(j)
                    cv2.putText(frame1, str(final), (450, x), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1,
                                lineType=2)
                    x += 25
                #cv2.putText(frame1, str(s1), (200, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), thickness=1,
                         #   lineType=2)
                cv2.putText(frame1, str("Model:"+t[-1]), (75, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1,
                            lineType=2)
                cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
                cv2.imshow('Video', frame1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        video_capture.release()
        cv2.destroyAllWindows()
