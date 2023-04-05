import cv2
import mediapipe as mp
import time
import os
import math

cap = cv2.VideoCapture(0)

wCam, hCam = 650, 650 
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

# Load models for hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

def draw_green_circle(x, y, img):
    cv2.circle(img=img, center=(x,y), radius=10, color=(0,255,0), thickness=cv2.FILLED)

while True:
    ### Read frame
    success, img = cap.read()
    ### Convert the frame image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ### Process the image and track hands
    results = hands.process(imgRGB)
    ### Print landmark connections if MULTIPLE hands are detected
    if results.multi_hand_landmarks:
        ### For each of the multiple hands
        for handLms in results.multi_hand_landmarks:
            ### Add the identified landmark id and the refactored position (in pixels)
            positions = {}
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                positions[id] = [cx, cy]
            # Draw line connections between landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            tipIds = {4:0, 8:6, 12:10, 16:14, 20:18}
            tipNames = {8:'index', 12:'middle', 16:'ring', 20:'pinky'}
            tipStr = "Open:"
            counter = 0
            for i in tipIds:
                ### Process thumb
                if i == 4:
                    x1, x2 = positions[0][0], positions[2][0]
                    y1, y2 = positions[0][1], positions[2][1]
                    reference = math.hypot(x2-x2, y2-y1)
                    x1, x2 = positions[i][0], positions[tipIds[i]][0]
                    y1, y2 = positions[i][1], positions[tipIds[i]][1]
                    length = math.hypot(x2-x1, y2-y1)
                    if reference != 0:
                        cv2.putText(img, f"{length/reference}", (40,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
                        if length/reference > 3:
                            counter += 1
                            tipStr += f" > thumb;"
                            draw_green_circle(positions[4][0], positions[4][1], img)
                ### Process all the other fingers:
                else:
                    if positions[i][1] < positions[tipIds[i]][1]:
                        counter += 1
                        tipStr += f" > {tipNames[i]};"
                        draw_green_circle(positions[i][0], positions[i][1], img)
            cv2.putText(img, f"Total open: {counter}", (40,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
            cv2.putText(img, tipStr, (40,80), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 3)

    # Display the captured frame image
    cv2.imshow("Hand tracking", img)
    # Display FPS on the screen
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    # Display the captured frame image
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video camera and close the window
cap.release()
cv2.destroyAllWindows()
