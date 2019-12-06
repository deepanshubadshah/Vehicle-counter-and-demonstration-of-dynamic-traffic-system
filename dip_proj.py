import cv2
import numpy as np
from time import sleep

width_min=40 #Minimum width of rectangle
height_min=40 #Minimum height of rectangle

offset=3 #Pixel error Allowed

pos_line=670 #Count-line position
entry_pos_line=450 #Count-line position

detec = []
carsin= 0
carsout= 0

cap = cv2.VideoCapture('video.mp4')
subtraction = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    tempo = float(1/60)                                 #Video Frames per sec.
    sleep(tempo)
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    cv2.imshow("Blur",blur)
    img_sub = subtraction.apply(blur)
    cv2.imshow("Subtracted",img_sub)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    cv2.imshow("Dilate",dilat)
    strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , strel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , strel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , strel)
    cv2.imshow("Dilated",dilated)

    contour,h = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1, (115, entry_pos_line), (660, entry_pos_line), (255,127,0), 3)
    cv2.line(frame1, (20, pos_line), (600, pos_line), (255,127,0), 3)

    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_contour = (w >= width_min) and (h >= height_min)
        if not validate_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        detec.append(((x+int(w/2)),(y+int(h/2))))
        cv2.circle(frame1, ((x+int(w/2)),(y+int(h/2))), 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(entry_pos_line+offset) and y>(entry_pos_line-offset) and x<(600):
                carsin+=1
                if((carsin-carsout)<0):
                    carsout = carsin
                cv2.line(frame1, (115, entry_pos_line), (600, entry_pos_line), (0,127,255), 3)
                print("No. of cars entered: "+str(carsin))
                print("No. of cars in frame: "+str(carsin-carsout))
                detec.remove((x,y))

            if (y<(pos_line+offset+2) and y>(pos_line-offset-2) and x<(600)) or x<(20):
                carsout= carsout+1
                if((carsin-carsout)<0):
                    carsout = carsin
                cv2.line(frame1, (10, pos_line), (600, pos_line), (0,127,255), 3)
                print("No. of cars exited: "+str(carsout))
                print("No. of cars in frame: "+str(carsin-carsout))
                detec.remove((x,y))

    cv2.putText(frame1, "Vehicles in: "+str(carsin), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
    cv2.putText(frame1, "Vehicles: "+str(carsin-carsout)+" (time="+str(15*(carsin-carsout))+")", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
    cv2.putText(frame1, "Vehicles out: "+str(carsout), (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
    cv2.imshow("Original Video" , frame1)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
