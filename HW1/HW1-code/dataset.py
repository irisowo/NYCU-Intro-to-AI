import os
import re
import numpy as np
import cv2
from cv2 import cv2 as cv
import random
from copy import deepcopy
import matplotlib.pyplot as plt

def rotate(image, angle=5, scale=1.0):
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv.warpAffine(image,M,(w,h))
    return image

#=======================================================================
def loadImages(dataPath,byteorder='<'):
    # Begin your code (Part 1)
    dataset = []
    for class_dir_name in ["face","non-face"]:
        class_dir = os.path.join(dataPath, class_dir_name)
        images_in_class = os.listdir(class_dir)
        for image_file in images_in_class:
          with open(class_dir+'/'+image_file, 'rb') as f:
              buffer_ = f.read()
          
          try:
            header, width, height, maxval = re.search(
              b"(^P5\s(?:\s*#.*[\r\n])*"
              b"(\d+)\s(?:\s*#.*[\r\n])*"
              b"(\d+)\s(?:\s*#.*[\r\n])*"
              b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer_).groups()
          except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % image_file)

          label= 1 if (class_dir_name=="face") else 0
          img = np.frombuffer(buffer_,dtype='u1' if int(maxval) < 256 else byteorder+'u2',count=int(width)*int(height),offset=len(header)).reshape(19, 19)
          dataset.append((img,label))
    return dataset

