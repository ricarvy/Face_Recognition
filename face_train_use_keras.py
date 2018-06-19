# -*- coding: utf-8 -*-
# @Time    : 2018/1/13 14:24
# @Author  : Li Jiawei
# @FileName: face_train_use_keras.py
# @Software: PyCharm

import random

from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import os
import cv2
import numpy as np

IMAGE_SIZE=64

def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
    top,bottom,left,right=(0,0,0,0)

    h,w,_=image.shape

    longest_edge=max(h,w)

    if h<longest_edge:
        dh=longest_edge-h
        top=dh//2
        bottom=dh-top
    elif w<longest_edge:
        dw=longest_edge-w
        left=dw//2
        right=dw=dw-left
    else:
        pass
    BLACK=[0,0,0]
    constant=cv2.copyMakeBorder(image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=BLACK)
    return cv2.resize(constant,(64,64))

images=[]
labels=[]
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path=os.path.abspath(os.path.join(path_name,dir_item))
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                img=cv2.imread(full_path)
                image=resize_image(img,IMAGE_SIZE,IMAGE_SIZE)

                images.append(image)
                labels.append(path_name)
    return images,labels

def load_dataset(path_name):
    images,labels=read_path(path_name)

    images=np.array(images)

    labels=np.array([0 if label.endswith('me') else 1 for label in labels])

    return images,labels

class Dataset:
    def __init__(self,path_name):
        self.train_images=None
        self.train_labels=None

        self.valid_images=None
        self.valid_labels=None

        self.test_images=None
        self.test_labels=None

        self.path_name=path_name

        self.input_shape=None

    def load(self,
             img_rows=IMAGE_SIZE,
             img_cols=IMAGE_SIZE,
             img_channels=3,
             nb_classes=2):
        images,labels=load_dataset(self.path_name)
        train_images,valid_images,train_labels,valid_labels=train_test_split(images,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=random.randint(0,100))
        _,test_images,_,test_labels=train_test_split(images,
                                                     labels,
                                                     test_size=0.3,
                                                     random_state=random.randint(0,100))
        if K.image_dim_ordering() == 'th':
            train_images=train_images.reshape(train_images.shape[0],
                                              img_channels,
                                              img_rows,
                                              img_cols)
            valid_images=valid_images.reshape(valid_images.shape[0],
                                              img_channels,
                                              img_rows,
                                              img_cols)
            test_images=test_images.reshape(test_images.shape[0],
                                            img_channels,
                                            img_rows,
                                            img_cols)
            self.input_shape=(img_channels,img_rows,img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0],
                                                img_rows,
                                                img_cols,
                                                img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0],
                                                img_rows,
                                                img_cols,
                                                img_channels)
            test_images = test_images.reshape(test_images.shape[0],
                                              img_rows,
                                              img_cols,
                                              img_channels)
            self.input_shape = (img_rows,
                                img_cols,
                                img_channels)
        print(train_images.shape)
        # print(valid_images.shape[0])
        # print(test_images.shape[0])

        train_labels=np_utils.to_categorical(train_labels,nb_classes)
        valid_labels=np_utils.to_categorical(valid_labels,nb_classes)
        test_labels=np_utils.to_categorical(test_labels,nb_classes)

        train_images=train_images.astype('float32')
        valid_images=valid_images.astype('float32')
        test_images=test_images.astype('float32')

        train_images/=255
        valid_images/=255
        test_images/=255

        self.train_images=train_images
        self.valid_images=valid_images
        self.test_images=test_images
        self.train_labels=train_labels
        self.valid_labels=valid_labels
        self.test_labels=test_labels

class Model:
    def __init__(self):
        self.model=None

    def build_model(self,dataset,nb_classes=2):
        self.model=Sequential()
        self.model.add(Convolution2D(32,3,3,border_mode='same',input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(32,3,3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))

        self.model.add(Activation('relu'))

        self.model.summary()
    def train(self,dataset,batch_size=20,nb_epoch=10,data_augmentation=False):
        sgd=SGD(lr=0.01,
                decay=1e-6,
                momentum=0.9,
                nesterov=True)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
        if not data_augmentation:
            print(dataset.train_images.shape,dataset.train_labels.shape,'shape')
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           epochs=nb_epoch,
                           validation_data=(dataset.valid_images,dataset.valid_labels))
        else:
            datagen=ImageDataGenerator()
            datagen.fit(dataset.train_images)
            self.model.fit_generator(datagen.flow(dataset.train_images,
                                                  dataset.train_labels,
                                                  batch_size=batch_size),
                                     batch_size=batch_size,
                                     sample_per_epoch=dataset.train_images.shape[0],
                                     nb_epoch=nb_epoch,
                                     validation_data=(dataset.valid_images,dataset.valid_labels))
    def save_model(self,file_path):
        self.model.save(file_path)
    def load_model(self,file_path):
        self.model=load_model(file_path)
    def evaluate(self,dataset):
        score=self.model.evaluate(dataset.test_images,dataset.test_labels,verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
    def face_predict(self,image):
        if K.image_dim_ordering() == 'th' and image.shape!=(1,3,IMAGE_SIZE,IMAGE_SIZE):
            image=resize_image(image)
            image=image.reshape((1,3,IMAGE_SIZE,IMAGE_SIZE))
        elif K.image_dim_ordering() == 'tf' and image.shape!=((1,IMAGE_SIZE,IMAGE_SIZE,3)):
            image=resize_image(image)
            image=image.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        image=image.astype('float32')
        image/=255

        result=self.model.predict_proba(image)
        print('result',result)

        result=self.model.predict_classes(image)
        return result[0]

# if __name__ == '__main__':
dataset=Dataset('data/')
dataset.load()
print('shape',dataset.train_images.shape,dataset.train_labels.shape)

    # # train
    # model=Model()
    # model.build_model(dataset=dataset)
    # model.train(dataset=dataset)
    # model.save_model(file_path='model/me.face.model.h5')

    # evaluate
    # model=Model()
    # model.load_model(file_path='model/me.face.model.h5')
    # model.evaluate(dataset)