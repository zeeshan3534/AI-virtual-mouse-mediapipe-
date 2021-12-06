import cv2
import mediapipe as mp
import time
#for video cam
cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands=mpHands.Hands()#is funciton ke backhand me ek image_static_mode leta jo ke detect krta , or maxumum hand leta hai ke kitne krne or detection range and tracking confidence
mpDraw = mp.solution.drawing_utils

ptime = 0
ctime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #for extracting multiple hands
    if results.multi_hand_landmarks:
        #this is for single hand
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                print(id,lm)
            #its draw a 21 points in hand AND CONNECTIONS
            mpDraw.draw_landmarks(img, handLms,mpHands.HAND_CONNECTIONS)


    ctime = time.time
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10,70),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)


    cv2.imshow("image",img)
    cv2.waitkey(1)