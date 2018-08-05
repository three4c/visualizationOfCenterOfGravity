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
    bpoints = gpoints = rpoints = ypoints = [deque(maxlen=512)]
    bindex = gindex = rindex = yindex = 0
    colors = [(0, 255, 0), (0, 0, 255)]

    csvHeader = ['X Coordinate', 'Y Coordinate']
    csvBodyRotation = []
    csvBodyCentroid = []

    cap = cv2.VideoCapture(0)
    xBody = []
    xAbs = 0
    l = 0

    while(True):
        ret, frame = cap.read()

        if ret == True:
            frame = cv2.flip(frame, 1)
            rectsGreen = colorTracking(frame, Green())

            if len(rectsGreen) > 0:
                height = [0, 0]
                rx, ry, rw, rh = max(rectsGreen, key=(lambda x: x[2] * x[3]))
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
                cv2.circle(frame, (int(rx + rw / 2), int(ry + rh / 2)), 5, (0,0,255), -1)
                center = (int(rx + rw / 2), int(ry + rh / 2))
                bpoints[bindex].appendleft(center)
                csvBodyRotation.append(center)
                xBody.append(int(rx + rw / 2))
                xAbs += abs(xBody[l-1] - xBody[l])
                height = (xAbs, int(ry + rh / 2))
                csvBodyCentroid.append(height)
                l += 1

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

            for i in range(len(points)):
                for j in range(len(points[i])):
                    xFront = 0
                    xBack = 0
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
                csvBodyRotation = csvBodyCentroid = xBody = []
                xAbs = l = 0
            if key == ord("s"):
                cv2.imwrite("photo.jpg", frame)

        else: break

    with open('bodyRotation.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(csvHeader)
            writer.writerows(csvBodyRotation)

    with open('bodyCentroid.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(csvHeader)
            writer.writerows(csvBodyCentroid)

    cap.release()
    cv2.destroyAllWindows()
