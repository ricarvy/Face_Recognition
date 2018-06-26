# -*- coding: utf-8 -*-
# @Time    : 2018/1/16 13:24
# @Author  : Li Jiawei
# @FileName: face_predict_use_keras_video.py
# @Software: PyCharm

import cv2
import sys
import gc
from face_train_use_keras import Model

model=Model()
model.load_model(file_path='model/me.face.model.h5')

color=(0,255,0)
cap=cv2.VideoCapture(0)

cascade_path='xml/haarcascade_frontalface_alt.xml'

while True:
    _,frame=cap.read()
    frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cascade=cv2.CascadeClassifier(cascade_path)

    faceRects=cascade.detectMultiScale(frame_gray,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
    if len(faceRects) >0:
        for faceRect in faceRects:
            x,y,w,h=faceRect
            image=frame[y-10:y+h+10,x-10:x+w+10]
            faceID=model.face_predict(image)

            if faceID == 0:
                cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,thickness=2)
                cv2.putText(frame,'men',(x+30,y+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            else:
                pass
    cv2.imshow('Recognization',frame)

    k=cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()