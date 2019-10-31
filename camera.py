# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 17:09:53 2019

@author: Otto
"""
import cv2
import numpy as np

def draw_box(x,y,img, size):
    img = cv2.line(img, (x,y), (x+size, y+0), (255,0,0),2) #image, pos1, pos2, colour, thickness
    img = cv2.line(img, (x+size,y), (x+size, y+size), (255,0,0),2)
    img = cv2.line(img, (x,y+size), (x+size, y+size), (255,0,0),2)
    img = cv2.line(img, (x,y), (x, y+size), (255,0,0),2)
    
    return img



cap = cv2.VideoCapture(0)
# if(cap.isOpened()== False):
#         exit(-1);
size = 224   

while(True):
    
    
    # Capture frame-by-frame
    ret, frame = cap.read()
   
    
    if ret == True:
        
        frame = cv2.flip(frame, 1)
        
        # if type(frame ==  'NoneType'):
            
        #     frame = np.zeros((512,512,3), np.uint8)
        
       
        frame = draw_box(int(frame.shape[1]/2 - size/2), int(frame.shape[0]/2  - size/2), frame, size)
    
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
    else:
        frame = np.zeros((512,512,3), np.uint8)
        cv2.imshow('frame', frame)
        
    k = cv2.waitKey(1)
    if (k== ord('q')):
        break
    elif (k == ord('s')):
        cv2.imwrite('image.bmp', frame)
    elif (k == ord('c')):
        x = int(frame.shape[1]/2 - size/2)
        y = int(frame.shape[0]/2  - size/2)
        #print(x, x+size, y, y+size)
        frame = frame[y:y+size , x:x+size]
        cv2.imwrite('image-crop.bmp', frame)
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()