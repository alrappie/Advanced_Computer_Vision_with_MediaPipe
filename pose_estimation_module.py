import cv2
import mediapipe as mp
import time

class Pose_Detector():
    
    def __init__(self,STATIC_IMAGE_MODE=False,MODEL_COMPLEXITY=1,SMOOTH_LANDMARKS=True,ENABLE_SEGMENTATION=False,SMOOTH_SEGMENTATION=True,MIN_DETECTION_CONFIDENCE=0.5,MIN_TRACKING_CONFIDENCE=0.5):
        self.static_image_mode = STATIC_IMAGE_MODE
        self.model_complexity = MODEL_COMPLEXITY
        self.smooth_landmarks = SMOOTH_LANDMARKS
        self.enable_segmentation = ENABLE_SEGMENTATION
        self.smooth_segmentation = SMOOTH_SEGMENTATION
        self.min_detection_confidence = MIN_DETECTION_CONFIDENCE
        self.min_tracking_confidence = MIN_TRACKING_CONFIDENCE
        
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,self.model_complexity,self.smooth_landmarks,self.enable_segmentation,self.smooth_segmentation,self.min_detection_confidence,self.min_tracking_confidence)


    def find_pose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def find_position(self,img,draw=True):       
        lm_list = []
        if self.results.pose_landmarks: 
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx , cy = int(lm.x*w), int(lm.y*h)
                lm_list.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lm_list
            
            

def main():
    cap = cv2.VideoCapture(r"\C:\Users\alrav\Videos\vid2.mp4")
    pTime = 0
    cTime = 0
    detector = Pose_Detector()
    while True:
        success,img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img,draw=False)
        cv2.circle(img,(lm_list[14][1],lm_list[14][2]),5,(255,0,0),cv2.FILLED)
        # print(lm_list)

        cTime = time.time()   
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),3)
    
        cv2.imshow("Image",img)
        cv2.waitKey(1)

    
if __name__ == '__main__':
    main()