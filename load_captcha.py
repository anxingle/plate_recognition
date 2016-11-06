#!/usr/bin/env python
# coding=utf-8

import numpy as np
import struct
import os
import cv2
import random

ROW   = 80
COL   = 120
depth = 1


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[[index_offset + labels_dense.ravel()]] = 1
  return labels_one_hot
def read_images(train_dir):
    images = os.listdir(train_dir)
    # volume of data_set (train_dir)
    data_volume = len(images)
    # get a random list intend to shuffle the data_set 
    images_list = range(data_volume)
    random.shuffle(images_list)
    random.shuffle(images_list)
    # return to the main Call function
    # data.shape  = (volume,row,col,depth)
    data  = np.zeros((data_volume,ROW*COL))
    # label.shape = (volume,row,col,depth)
    #label = np.zeros((data_volume,1))
    #label = np.zeros((data_volume,1))
    #当label长度不是5的时候
    #label = np.zeros((data_volume,3))
    # I want to chang to list
    label  = []
    for loop in xrange(data_volume):
        image_name  =  images[images_list[loop]]
        image       =  os.path.join(train_dir,image_name)
        img         =  cv2.imread(image,cv2.IMREAD_GRAYSCALE)
        #XXX: I want to feed data by COL
        #XXX: 按列来组织图片数据

        img         =  img.transpose(1,0)
        img         =  img.reshape((ROW*COL))
        #img         =  img.reshape((ROW,COL,depth))
        data[loop]  =  img

        image_name  =  image_name.strip()
        image_name  =  image_name.split('.')[0]
        image_name  =  image_name.split('_')[1]
        image_label =  image_name[:]
        #当label长度不是5的时候
        
        image_label = []
        length_name = len(image_name)
        for i in xrange(length_name):
            image_label.append(image_name[i])
        if  length_name != 3:
            for j in xrange(length_name,3):
                image_label[i] = 10
        label.append(image_label)
        
        #label[loop] = image_label
    label_num = np.array(label)
    return data,label_num


def read_data(train_dir,one_hot=False,reshape=True):
#def read_data(train_dir,one_hot=True,dtype=dtypes.float32,reshape=True):
    train_images,train_labels = read_images(train_dir)
    if one_hot:
        train_labels = dense_to_one_hot(train_labels,num_class)
    return train_images,train_labels
