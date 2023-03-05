import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(r"\C:\Users\alrav\Videos\vid1.mp4")
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face_detection = mpFaceDetection.FaceDetection()


while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgRGB)
    
    if results.detections:
        for id,detection in enumerate(results.detections):
            # print(id,detection)
            # print(detection.location_data.relative_bounding_box)
            
            # mpDraw.draw_detection(img,detection)
            
            ## manual draw
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)
            
            
            
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS : {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    # result = 
    cv2.imshow('image',img)
    cv2.waitKey(1)