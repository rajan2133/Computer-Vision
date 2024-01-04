#-----------------Low Pass------------------------------
import cv2
import numpy
import math
img = cv2.imread('C:/Users/Dell/OneDrive/Documents/Computer Vision/LAB/orignal_img.png')
cv2.imshow('orignal image',img)


filter_1 = numpy.ones((3,3)) / 9 
img_filter_1 = cv2.filter2D(src=img,ddepth=(-1),kernel=filter_1)
cv2.imshow('img_filter_3x3_9 ',img_filter_1)
                            
        
filter_2 = numpy.ones((5,5),numpy.float32) / 25 
img_filter_2 = cv2.filter2D(src=img,ddepth=(-1),kernel=filter_2)
cv2.imshow('img_filter_5x5_25 ',img_filter_2)


filter_3 = numpy.ones((11,11),numpy.float32) / 121
img_filter_3 = cv2.filter2D(src=img,ddepth=(-1),kernel=filter_3)
cv2.imshow('img_filter_11x11_121 ',img_filter_3)
     

filter_4 = numpy.ones((5,5),numpy.float32) / 30
img_filter_4 = cv2.filter2D(src=img,ddepth=(-1),kernel=filter_4)
cv2.imshow('img_filter_5x5_30 ',img_filter_4)
cv2.waitKey(0)
cv2.destroyAllWindows()


#--------------------High Pass-------------------------

import cv2
import numpy as np
ker = np.array([[0,-1,0],
[-1,5,-1],
[0,-1,0]])

ker_2 = np.array([[0,-1,0],
[-1,4,-1],
[0,-1,0]])

#img = cv2.imread('blured_1.png')
img = cv2.imread('C:/Users/Dell/OneDrive/Documents/Computer Vision/LAB/orignal_img.png')
img_high_pass = cv2.filter2D(img,-1,ker) 
img_high_pass_2 = cv2.filter2D(img,-1,ker_2) 
cv2.imshow("orignal image ",img) 
cv2.imshow("High_1",img_high_pass)
 
cv2.imshow("High_2",img_high_pass_2) 
cv2.waitKey(0) 
cv2.destroyAllWindows()











































