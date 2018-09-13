import numpy as np
import cv2
import csv
from collections import deque

# Color
class Red:
    def __init__(self):
        self.lower = np.array([150, 50, 50])
        self.upper = np.array([180, 255, 255])

class Blue:
    def __init__(self):
        self.lower = np.array([100, 60, 60])
        self.upper = np.array([140, 255, 255])

class Green:
    def __init__(self):
        self.lower = np.array([40, 50, 50])
        self.upper = np.array([80, 255, 255])

# Function
def colorTracking(frame, colorObj):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colorObj.lower, colorObj.upper)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    (_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    rects = []
    for contour in cnts:
        convexHull = cv2.convexHull(contour)
        boundingRect = cv2.boundingRect(convexHull)
        rects.append(np.array(boundingRect))

    return rects

# Main
if __name__ == '__main__':
    kernel = np.ones((5, 5), np.uint8)
    bpoints01 = gpoints01 = rpoints01 = ypoints01 = [deque(maxlen=512)]
    bindex01 = gindex01 = rindex01 = yindex01 = 0
    bpoints02 = gpoints02 = rpoints02 = ypoints02 = [deque(maxlen=512)]
    bindex02 = gindex02 = rindex02 = yindex02 = 0
    colors = [(0, 255, 0), (0, 0, 255)]

    csvUpperBodyTrajectory = []
    csvWaistTrajectory = []

    cap01 = cv2.VideoCapture(1)
    cap02 = cv2.VideoCapture(2)

    xBody = []
    xAbs = 0
    l = 0

    while(True):
        _, frame01 = cap01.read()
        frame01 = cv2.flip(frame01, 1)
        rectsGreen01 = colorTracking(frame01, Green())

        _, frame02 = cap02.read()
        frame02 = cv2.flip(frame02, 1)
        rectsGreen02 = colorTracking(frame02, Green())

        if len(rectsGreen01) > 0 and len(rectsGreen02) > 0:
            rx, ry, rw, rh = max(rectsGreen01, key=(lambda x: x[2] * x[3]))
            cv2.rectangle(frame01, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
            cv2.circle(frame01, (int(rx + rw / 2), int(ry + rh / 2)), 5, (0,0,255), -1)
            center01 = (int(rx + rw / 2), int(ry + rh / 2))
            bpoints01[bindex01].appendleft(center01)
            csvUpperBodyTrajectory.append(center01)

            height = [0, 0]
            lx, ly, lw, lh = max(rectsGreen02, key=(lambda x: x[2] * x[3]))
            cv2.rectangle(frame02, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 3)
            cv2.circle(frame02, (int(lx + lw / 2), int(ly + lh / 2)), 5, (0,0,255), -1)
            center02 = (int(lx + lw / 2), int(ly + lh / 2))
            bpoints02[bindex02].appendleft(center02)
            xBody.append(int(lx + lw / 2))
            xAbs += abs(xBody[l-1] - xBody[l])
            height = (xAbs, int(ly + lh / 2))
            csvWaistTrajectory.append(height)
            l += 1

        else:
            bpoints01.append(deque(maxlen=512))
            bindex01 += 1
            gpoints01.append(deque(maxlen=512))
            gindex01 += 1
            rpoints01.append(deque(maxlen=512))
            rindex01 += 1
            ypoints01.append(deque(maxlen=512))
            yindex01 += 1

            bpoints02.append(deque(maxlen=512))
            bindex02 += 1
            gpoints02.append(deque(maxlen=512))
            gindex02 += 1
            rpoints02.append(deque(maxlen=512))
            rindex02 += 1
            ypoints02.append(deque(maxlen=512))
            yindex02 += 1

        points01 = [bpoints01, gpoints01, rpoints01, ypoints01]
        points02 = [bpoints02, gpoints02, rpoints02, ypoints02]

        for i in range(len(points01)):
            for j in range(len(points01[i])):
                for k in range(1, len(points01[i][j])):
                    if points01[i][j][k - 1] is None or points01[i][j][k] is None: continue
                    # Circle
                    cv2.line(frame01, points01[i][j][k - 1], points01[i][j][k], colors[0], 2)

        for i in range(len(points02)):
            for j in range(len(points02[i])):
                xFront = 0
                xBack = 0
                for k in range(1, len(points02[i][j])):
                    if points02[i][j][k - 1] is None or points02[i][j][k] is None: continue

                    # Line graph
                    front = points02[i][j][k - 1]
                    back = points02[i][j][k]
                    xBack += abs(front[0] - back[0])
                    cv2.line(frame02, (xFront, front[1]), (xBack, back[1]), colors[1], 2)
                    xFront = xBack

        cv2.namedWindow("Upper body trajectory", cv2.WINDOW_NORMAL)
        cv2.imshow("Upper body trajectory", frame01)

        cv2.namedWindow("Waist trajectory", cv2.WINDOW_NORMAL)
        cv2.imshow("Waist trajectory", frame02)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap01.release()
            cap02.release()
            cv2.destroyAllWindows()
            break

        if key == ord("w"):
            bpoints01 = gpoints01 = rpoints01 = ypoints01 = [deque(maxlen=512)]
            bindex01 = gindex01 = rindex01 = yindex01 = 0

            bpoints02 = gpoints02 = rpoints02 = ypoints02 = [deque(maxlen=512)]
            bindex02 = gindex02 = rindex02 = yindex02 = 0

            csvUpperBodyTrajectory = []
            csvWaistTrajectory = []

            xBody = []
            xAbs = 0
            l = 0

        if key == ord("s"):
            cv2.imwrite("photo01.jpg", frame01)
            cv2.imwrite("photo02.jpg", frame02)

    with open('UpperBodyTrajectory.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(csvUpperBodyTrajectory)

    with open('WaistTrajectory.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(csvWaistTrajectory)
