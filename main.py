# importing dependencies
import cv2
import mediapipe as mp
import numpy as np
import mouse
import time

# opening camera
cap = cv2.VideoCapture(0)

# hand recognition module intialisation
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
Hand = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
px, py , cx, cy = 0, 0, 0, 0
# screen values
scrnw, scrnh = 1920, 1080
# Landmarking 

def handLandmark(Cimg):
    landmarkList = []
    landmarkPositions = Hand.process(Cimg)
    landmarkCheck = landmarkPositions.multi_hand_landmarks
    # checking tracked landmarks
    if landmarkCheck:
        for hand in landmarkCheck:
            for index,landmark in enumerate(hand.landmark):
                mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS)
                h, w, c = img.shape
                X, Y = int(landmark.x * w), int(landmark.y*h)
                landmarkList.append([index, X, Y])
    return landmarkList


def fingers(landmarks):
    ftips = []
    tipid = [4, 8, 12, 16, 20]
    # for thumb
    if landmarks[tipid[0]][1] > lmlist[tipid[0] - 1][1]:
        ftips.append(1)
    else:
        ftips.append(0)
    # for other fingers
    for id in range(1, 5):
        # Checks to see if the tip of the finger is higher than the joint
        if landmarks[tipid[id]][2] < landmarks[tipid[id] -3][2]:
            ftips.append(1) # finger opened
        else:
            ftips.append(0) # finger closed 

    return ftips

while True:
    check, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lmlist = handLandmark(imgRGB)
    if len(lmlist) != 0:
        # Getting fingertips values
        finger = fingers(lmlist)
        # creating a realtion between cv screen and window screen
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]
        x3 = np.interp(x1, (75, 640-75), (0, scrnw))
        y3 = np.interp(y1, (75, 480-75), (0, scrnh))
        # checking for pointer
        if finger[1] == 1 and finger[2] == finger[3] == finger[4] == 0:
            # movement smoothness
            cx = px + (x3-px)/10
            cy = py + (y3-py)/10
            mouse.move(scrnw-x3,y3,absolute=True) # inverting
            px, py = cx, cy
        # checking for clicking
        if finger[1] ==1 and finger[2] == 1:
            mouse.click(button="left")
            time.sleep(0.2)
    cv2.imshow("Hand Tracking",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing resources
cap.release()
cv2.destroyAllWindows()