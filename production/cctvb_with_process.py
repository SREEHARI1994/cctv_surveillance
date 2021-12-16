# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:15:05 2018

@author: user39
"""

# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import Jetson.GPIO as GPIO
import time
import face_recognition
import os
from multiprocessing import Process
#from playsound import playsound

#from threading import Thread

def alarm():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setwarnings(True)
	GPIO.setup(23,GPIO.OUT)
	try:
		print ("LED on")
		GPIO.output(23,GPIO.HIGH)
		time.sleep(5)

        #print ("LED off")
		GPIO.output(23,GPIO.LOW)
		GPIO.cleanup()
		p.terminate()
	except:
		KeyboardInterrupt


obama_image = face_recognition.load_image_file("/home/sreehari/Desktop/project/hashim.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("/home/sreehari/Desktop/project/sree.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Hashim",
    "Sreehari"
]



def face_detection(fbox,frame):
	print("doing face detection")
	#cv2.imshow("full_image",frame)
	p = Process(target=alarm)
	x1=fbox[1]
	x2=fbox[3]
	y1=fbox[0]
	y2=fbox[2]
	x1=int((1280/300)*x1)
	x2=int((1280/300)*x2)
	y1=int((720/300)*y1)
	y2=int((720/300)*y2)
	crop=frame[y1:x2,x1:x2]
	#cv2.imshow("face",crop)
	rgb_small_frame = crop[:, :, ::-1]
	#cv2.imshow("face",rgb_small_frame)
	#rgb_small_frame=frame
	face_locations = face_recognition.face_locations(rgb_small_frame)
	face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

	face_names = []
	
    
	for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
		name = "unknown"

            # If a match was found in known_face_encodings, just use the first one
		if True in matches:
		#if not (True in matches):
			#print("sreehari")
				
			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]

                
		else:
			print("BURGLER DETECTED")
			if (not (p.is_alive())):
				p = Process(target=alarm)
				p.start()

			

            
            #playsound('/home/arvind/Desktop/cut.wav')
            
        #if not face_encodings:
            #print("masked man detected")  
    
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):






        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
	p = Process(target=alarm)
   
    
	model_path = '/home/sreehari/Desktop/project/v2coco2722/frozen_inference_graph.pb'
	#model_path = '/home/sreehari/Desktop/project/resnet7635/frozen_inference_graph.pb'
	odapi = DetectorAPI(path_to_ckpt=model_path)
	threshold = 0.7
	#cap = video_capture 
	ni=0
	while True:
		if ni%12==0:
			video_capture=cv2.VideoCapture('rtsp://admin:admin1234@192.168.1.3/ch1/main/av_stream')
			r, img = video_capture.read()
		#if img.any():
				#print ("got it")
			#else:
				#print ("get lost")
			img = cv2.resize(img, (1280, 720))
			rsimg = cv2.resize(img, (300, 300))
        #if ni%20==0:
			boxes, scores, classes, num = odapi.processFrame(rsimg)

        # Visualization of the results of a detection.

			for i in range(len(boxes)):
            # Class 1 represents human
				if classes[i] == 1 and scores[i] > threshold:
					print ("human detected")
					box = boxes[i]
				
				#face_detection(crop,img)
					face_detection(box,img)
				
                #write(crop)
                
               
				#cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                #ni=ni+20
            
       
       
		ni=ni+1
		
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
        
		#time.sleep(0.68958513)
    
                    
video_capture.release()
cv2.destroyAllWindows()

