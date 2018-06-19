# -*- coding: utf-8 -*-
# @Time    : 2018/1/12 19:09
# @Author  : Li Jiawei
# @FileName: dataPreparation.py
# @Software: PyCharm

import cv2

def catchPictureFromCamera(windows_name,device_index,catch_num,path_name):
    cv2.namedWindow(winname=windows_name)
    classifier=cv2.CascadeClassifier('../Algorithm/Object_Detection/data/haarcascade_frontalface_alt.xml')
    cap=cv2.VideoCapture(device_index)
    num=0
    while cap.isOpened():
        flag,frame=cap.read()
        if not flag:
            break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faceRects=classifier.detectMultiScale(gray,1.2,minNeighbors=3,minSize=(32,32))
        if len(faceRects) >0:
            for faceRect in faceRects:
                x,y,w,h=faceRect

                img_name = '%s/%d.jpg' % (path_name, num)
                image=frame[y-10:y+h+10,x-10:x+w+10]
                cv2.imwrite(img_name,image)

                num+=1
                if num>catch_num:
                    break
                cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
                font=cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)
        if num > catch_num:
                break
        cv2.imshow(windows_name,frame)
        c=cv2.waitKey(10)
        if c == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

catchPictureFromCamera('Image Capture',0,100,'data/me')