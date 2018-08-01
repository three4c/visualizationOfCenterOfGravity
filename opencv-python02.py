import numpy as np
import cv2
from collections import deque

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

kernel = np.ones((5, 5), np.uint8)
bpoints = gpoints = rpoints = ypoints = [deque(maxlen=512)]
bindex = gindex = rindex = yindex = 0
colors = [(0, 255, 0), (0, 0, 255)]

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        if ret == True:
            frame = cv2.flip(frame, 1)
            rectsGreen = colorTracking(frame, Green())

            if len(rectsGreen) > 0:
                # cnt = sorted(rectsGreen, key = cv2.contourArea, reverse = True)[0]
                # ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # M = cv2.moments(cnt)
                rx, ry, rw, rh = max(rectsGreen, key=(lambda x: x[2] * x[3]))
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
                cv2.circle(frame, (int(rx + rw / 2), int(ry + rh / 2)), 5, (0,0,255), -1)
                center = (int(rx + rw / 2), int(ry + rh / 2))
                bpoints[bindex].appendleft(center)

            else:
                bpoints.append(deque(maxlen=512))
                bindex += 1
                gpoints.append(deque(maxlen=512))
                gindex += 1
                rpoints.append(deque(maxlen=512))
                rindex += 1
                ypoints.append(deque(maxlen=512))
                yindex += 1

            points = [bpoints, gpoints, rpoints, ypoints]

            xFront = 0
            xBack = 0

            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None: continue

                        # Circle
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[0], 2)

                        # Line graph
                        front = points[i][j][k - 1]
                        back = points[i][j][k]
                        xBack += abs(front[0] - back[0])
                        cv2.line(frame, (xFront, front[1]), (xBack, back[1]), colors[1], 2)
                        xFront = xBack

            cv2.imshow("frame", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("w"):
                bpoints = gpoints = rpoints = ypoints = [deque(maxlen=512)]
                bindex = gindex = rindex = yindex = 0
            if key == ord("s"):
                cv2.imwrite("photo.jpg", frame)

        else: break

    cap.release()
    cv2.destroyAllWindows()
