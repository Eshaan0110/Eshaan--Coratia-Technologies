import cv2
import mediapipe as mp
import time
import math

def anglefinder(C, A=50, B=50):
    cos_a = ((-(A**2)) + (B**2) + (C**2)) / (2 * B * C)
    cos_b = ((A**2) + (-(B**2)) + (C**2)) / (2 * A * C)
    cos_c = ((A**2) + (B**2) + (-(C**2))) / (2 * A * B)

    alpha = math.degrees(math.acos(cos_a))
    beta = math.degrees(math.acos(cos_b))
    gamma = math.degrees(math.acos(cos_c))
    return [gamma, alpha, beta, C]

def anglecorrector(y, angle, maxangle=90, range=400):
    return ((y / range) * maxangle) + angle

def gripangle(pointA, pointB):
    y = int(distcalc(pointA, pointB, 1)) + 20
    return min(y, 130)

def bruhh(x):
    return int((((100 - x) / 60) * 90) + 125)

def planeanglefinder(pointA, pointB, pointC):
    x1, y1, z1 = pointA
    x2, y2, z2 = pointB
    x3, y3, z3 = pointC
    a1, b1, c1 = x2 - x1, y2 - y1, z2 - z1
    a2, b2, c2 = x3 - x1, y3 - y1, z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    cosa = (c * 1300) / (1300 * (a**2 + b**2 + c**2)**0.5)
    x = math.acos(cosa)
    angle = abs(((math.degrees(x) - 89) * 10000) - 9960)
    return int(angle)

def tilt(pointA, pointB):
    x = int(distcalc(pointA, pointB, 1))
    return min(x, 180)

def distcalc(p1, p2, sel=0):
    x2, y2, z2 = p2
    x1, y1, z1 = p1
    if sel == 1:
        z1 = z2 = 0
    return ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

initcordinate = [2, 33, 69]
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    try:
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy, cz = (int(lm.x * w) - 300) * -1, (int(lm.y * h) - 500) * -1, (lm.z * 5000) * -1
                    if id == 9:
                        coordinate = [cx, cy, cz]
                        yolo = anglefinder(distcalc(coordinate, initcordinate) / 5)
                        angle1 = int(yolo[0])
                        angle2 = int(anglecorrector(cy, yolo[1]))
                        angle3 = int(anglecorrector(cx, 0, 90, 300)) + 90
                    if id == 5: pointA = [cx, cy, cz]
                    if id == 17: pointB = [cx, cy, cz]
                    if id == 1: pointC = [cx, cy, cz]
                    if id == 4: pointgripA = [cx, cy, cz]
                    if id == 8: pointgripB = [cx, cy, cz]
                    if id == 12: pointextraB = [cx, cy, cz]

                angle4 = planeanglefinder(pointA, pointB, pointC)
                finalangle5 = gripangle(pointgripA, pointgripB)
                finalangle2 = bruhh(angle2)
                finalangle1 = max(angle1 - 15, 0)
                finalangle4 = tilt(pointgripA, pointextraB)

                print(f"Angle 1 (base tilt): {finalangle1}")
                print(f"Angle 2 (elevation): {finalangle2}")
                print(f"Angle 3 (yaw): {angle3}")
                print(f"Angle 4 (wrist): {finalangle4}")
                print(f"Angle 5 (grip): {finalangle5}")
                print("-" * 30)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", cv2.flip(img, 1))
        cv2.waitKey(1)

    except Exception:
        pass
