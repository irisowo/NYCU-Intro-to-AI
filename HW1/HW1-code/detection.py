import cv2
import utils
import numpy as np
from matplotlib import pyplot as plt
from cv2 import cv2 as cv


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """

 
    #info from txt
    image_name = ""
    face_cnt = 0
    faces_loc = []
    # line_state = 1 if it's a new image
    line_state = 1
    line_cnt = 0
    # read file
    file = open(dataPath,"r")
    for line in file:
      line = line.replace("\n", "")
      if (line_state == 1 ):
        line_cnt = 0
        faces_loc.clear()
        #read image name and face_cnt
        fields = line.split(" ")
        image_name=fields[0]
        face_cnt=(int)(fields[1])
        line_state = 2
      elif (line_state == 2):
        fields = line.split(" ")
        faces_loc.append([fields[0],fields[1],fields[2],fields[3]])
        line_cnt+=1
        if (line_cnt == face_cnt):
          image = cv.imread('data/detect/' + image_name)
          image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
          image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
          for face in faces_loc:
            # cut the image
            x=(int)(face[0])
            y=(int)(face[1])
            w=(int)(face[2])
            h=(int)(face[3])
            face_image = image_gray[y:y+h,x:x+w]
            resized_image = cv.resize(face_image, (19, 19), interpolation=cv.INTER_AREA)
            #inhance the image
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            inhanced_image = clahe.apply(resized_image)
            if ( clf.classify(resized_image) or clf.classify(inhanced_image) ):
              cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
            else:
              cv.rectangle(image, (x, y), (x+w, y+h), (255, 0,0), 4)
          plt.imshow(image)
          plt.show()
          line_state =1
    file.close()
