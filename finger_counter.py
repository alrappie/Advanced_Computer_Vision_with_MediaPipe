import cv2
import time
import numpy as np
import os
import hand_tracking_module as htm

WCAM, HCAM = 1280,720
PATH = "./Img/FingerImg/"

cap = cv2.VideoCapture(0)
cap.set(3,WCAM)
cap.set(4,HCAM)

myList = os.listdir(PATH)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{PATH}/{imPath}')
    image = cv2.resize(image,(200,200))
    overlayList.append(image)

# print(len(overlayList))
pTime = 0

detector = htm.handDetector(detection_conf=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img,draw=False)
    if len(lmList)!= 0:
        # print(lmList)
        fingers = []
        
        
        for id in tipIds:
            if id == 4:
                if lmList[id][1] < lmList[3][1]:
                    fingers.append(0)
                    # print('here')
                else:
                    fingers.append(1)
            elif lmList[id][2] < lmList[id-2][2]:
                # print('finger open')  
                fingers.append(1)   
            else:
                fingers.append(0)
                
                
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        h, w, c = overlayList[totalFingers].shape
        img[0:h,0:w] = overlayList[totalFingers]
        
        cv2.rectangle(img,(20,255),(170,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(350,50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2,(15,15,249),2) #B G R
    
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    