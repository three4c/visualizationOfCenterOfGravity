import numpy as np
import cv2
from collections import deque

class Green:
    def __init__(self):
        self.lower = np.array([40, 0, 0])
        self.upper = np.array([80, 255, 255])

def colorTracking(frame, colorObj):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blueMask = cv2.inRange(hsv, colorObj.lower, colorObj.upper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    return cnts

kernel = np.ones((5, 5), np.uint8)
bpoints = gpoints = rpoints = ypoints = [deque(maxlen=512)]
bindex = gindex = rindex = yindex = 0
colors = [(0, 255, 0), (255, 0, 0)]

if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)

            rectsGreen = colorTracking(frame, Green())

            if len(rectsGreen) > 0:
                cnt = sorted(rectsGreen, key = cv2.contourArea, reverse = True)[0]
                ((x, y), radius) = cv2.minEnclosingCircle(cnt)
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # 中心座標
                M = cv2.moments(cnt)
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
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
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[0], 2)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
