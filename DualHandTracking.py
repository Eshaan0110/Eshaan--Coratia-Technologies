import cv2
import mediapipe as mp
import time
import math

def distcalc(p1, p2, sel=0):
    x2, y2, z2 = p2
    x1, y1, z1 = p1
    if sel == 1:
        z1 = z2 = 0
    return ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

def anglefinder(C, A=50, B=50):
    try:
        cos_a = ((-(A**2)) + (B**2) + (C**2)) / (2 * B * C)
        cos_b = ((A**2) + (-(B**2)) + (C**2)) / (2 * A * C)
        cos_c = ((A**2) + (B**2) + (-(C**2))) / (2 * A * B)
        alpha = math.degrees(math.acos(cos_a))
        beta = math.degrees(math.acos(cos_b))
        gamma = math.degrees(math.acos(cos_c))
        return [gamma, alpha, beta, C]
    except:
        return [0, 0, 0, 0]

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
    try:
        cosa = (c * 1300) / (1300 * (a**2 + b**2 + c**2)**0.5)
        x = math.acos(cosa)
        angle = abs(((math.degrees(x) - 89) * 10000) - 9960)
        return int(angle)
    except:
        return 0

def tilt(pointA, pointB):
    x = int(distcalc(pointA, pointB, 1))
    return min(x, 180)

def calculate_hand_angles(landmarks):
    try:
        id_map = {id: (x, y, z) for id, x, y, z in landmarks}
        coordinate = id_map[9]
        initcordinate = [2, 33, 69]
        yolo = anglefinder(distcalc(coordinate, initcordinate) / 5)
        angle1 = int(yolo[0])
        angle2 = int(anglecorrector(coordinate[1], yolo[1]))
        angle3 = int(anglecorrector(coordinate[0], 0, 90, 300)) + 90
        pointA = id_map[5]
        pointB = id_map[17]
        pointC = id_map[1]
        gripA = id_map[4]
        gripB = id_map[8]
        extraB = id_map[12]
        angle4 = planeanglefinder(pointA, pointB, pointC)
        finalangle1 = max(angle1 - 15, 0)
        finalangle2 = bruhh(angle2)
        finalangle3 = angle3
        finalangle4 = tilt(gripA, extraB)
        finalangle5 = gripangle(gripA, gripB)
        return {
            "Base Tilt": finalangle1,
            "Elevation": finalangle2,
            "Yaw": finalangle3,
            "Wrist Tilt": finalangle4,
            "Grip": finalangle5
        }
    except:
        return None

# Main Loop
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            handType = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
            h, w, c = img.shape
            landmarks = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy, cz = (int(lm.x * w) - 300) * -1, (int(lm.y * h) - 500) * -1, int(lm.z * 5000) * -1
                landmarks.append((id, cx, cy, cz))

            angles = calculate_hand_angles(landmarks)
            if angles:
                print(f"{handType} Hand Angles:")
                for k, v in angles.items():
                    print(f"  {k}: {v}")
                print("-" * 40)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime - pTime != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", cv2.flip(img, 1))
    cv2.waitKey(1)
