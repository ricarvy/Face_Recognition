# -*- coding: utf-8 -*-

# @Time  :2018/6/26 0:51
# @File  : face_predict_use_keras_photos.py
# @Author: LI Jiawei

import cv2
import sys
import gc
from face_train_use_keras import Model

model=Model()
model.load_model(file_path='model/me.face.model.h5')
color=(0,255,0)
filename='data/m/3.jpg'
gender = filename.split('/')[1]
cascade_path='xml/haarcascade_frontalface_alt.xml'

frame = cv2.imread(filename=filename)
frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
cascade=cv2.CascadeClassifier(cascade_path)

faceRects=cascade.detectMultiScale(frame_gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
if len(faceRects) >0:
     for faceRect in faceRects:
        x,y,w,h=faceRect
        image=frame[y-10:y+h+10,x-10:x+w+10]
        faceID=model.face_predict(image)
        if faceID == 0:
            gender = 'm'
        else :
            gender = 'w'
        cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
        cv2.putText(frame, gender, (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
cv2.imshow('Recognization',frame)

cv2.waitKey()
cv2.destroyAllWindows()