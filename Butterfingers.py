import cv2
import mediapipe as mp
import pyautogui
import numpy as np
mp_drawing=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands
cap=cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
SMOOTHING = 1.7
previous_x, previous_y = 0, 0
SCALING_FACTOR=1.9

with mp_hands.Hands( static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,) as hands:
    
    while True:
        ret,frame=cap.read()
        a=0
        if ret == False:
            break
        h,w=frame.shape[:2]
        frame=cv2.flip(frame,1)
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hands.process(frame_rgb)
        disd=0
        disu=0
        disl=0
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                x0=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x*w)
                y0=int((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y-0.4)*h)
                x1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*w)
                y1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*h)
                x2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*w)
                y2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*h)
                x3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*w)
                y3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*h)
                x4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x*w)
                y4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y*h)
                disd=((x2-x1)2+(y2-y1)2)(1/2)
                disu=((x3-x1)2+(y3-y1)2)(1/2)
                disl=((x4-x1)2+(y4-y1)2)(1/2)
                screen_x = np.interp(x0, (0, w), (0, screen_width))
                screen_y = np.interp(y0, (0, h), (0, screen_height))
                
                target_x = min(screen_width-1,screen_width / 2 + SCALING_FACTOR * (screen_x - screen_width / 2))
                target_y = min(screen_height-1,screen_height / 2 + SCALING_FACTOR * (screen_y - screen_height / 2))

                current_x = min(screen_width-1,previous_x + (target_x - previous_x) / SMOOTHING)
                current_y = min(screen_height-1,previous_y + (target_y - previous_y) / SMOOTHING)
                


                #if disd>0 and disd<30:
                    #pyautogui.moveTo(a*2,b*2, duration=1)
                pyautogui.moveTo(current_x,current_y)
                previous_x, previous_y=current_x ,current_y
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
        if disd>0 and disd<25:
            pyautogui.leftClick()
            #pyautogui.moveTo(100, 100, duration=1)
        elif disu>0 and disu<30:
            pyautogui.scroll(-150)
        elif disl>0 and disl<30:
            pyautogui.scroll(150)
        cv2.putText(frame,f"Distance:{disd:.2f}",(20,20),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(frame,f"Distance:{disu:.2f}",(40,40),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(frame,f"Distance:{disl:.2f}",(60,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1)& 0xFF==27:
            break
cap.release()
cv2.destroyAllWindows()