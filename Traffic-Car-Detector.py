import cv2
import numpy as np
from time import sleep
width_min=80 #minimum width of the rectangle
hight_min=80 #minimum hight of the rectangle
offset=6 #allowed error pixel 
pos_line=550 #position of the line 
delay= 60 #number of the frames per second 
detect = []#detect recent car centroid array
cars= 0#cars counter
def centroid(x, y, w, h): #get the center of the object(vehicle)
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy
cap = cv2.VideoCapture('highway.mp4')#the recorded video
subtractor = cv2.createBackgroundSubtractorMOG2()#to subtract the background & do some advanced functions like 1-noise removal | 2- shadow detection ,etc.
while True:#to loop along with our record
    _, frame = cap.read()#read a frame from the recorded video
    temp = float(1/delay)#get frames recommended (60)
    sleep(temp)#get frames recommended (60)
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#turn the frame to gray level
    blur = cv2.GaussianBlur(grey,(3,3),5)#blur the frame for noise removal and smoothing the frame 
    img_sub = subtractor.apply(blur)#to subtract the frame and the background
    dilat = cv2.dilate(img_sub,np.ones((5,5))) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contour,_  = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#get the contours of the frame
    cv2.line(frame, (25, pos_line), (1200, pos_line), (255,127,0), 3)#draw the detect line on the recorded video
    for(i,j) in enumerate(contour):#loop
        (x,y,w,h) = cv2.boundingRect(j)#detect the minimal right rectangle on the object 
        valid_contour = (w >= width_min) and (h >= hight_min)
        if not valid_contour:
            continue
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)#draw the minimal right rectangle on the object
        center = centroid(x, y, w, h)#get the center of the object
        detect.append(center)#append to car centroid array
        cv2.circle(frame, center, 4, (255, 0,0), -1)#draw the centroid on the object
        for (x,y) in detect:#loop
            if y<(pos_line+offset) and y>(pos_line-offset):#if error pixel reached to offset (cut the line by any car)
                cars+=3#add one object to the counter
                cv2.line(frame, (25, pos_line), (1200, pos_line), (0,127,255), 3)#draw the cutted line  
                detect.remove((x,y))#remove the detected recent car centroid              
    cv2.putText(frame, "# Cars : "+str(cars), (800, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),2)
    cv2.putText(frame, "# Cars : "+str(cars), (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),2)
    cv2.putText(frame, "Created by FCAI Team ", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),1)
    #cv2.putText(frame, "RABE3A ", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),1)
    #cv2.putText(frame, "KHALIFA ", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),1)
    cv2.imshow("Video Original" , frame)#the original needed camera vision
    cv2.imshow("Gaussian",blur)
    cv2.imshow("Detection",dilatada)#output the camera vision needed
    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()