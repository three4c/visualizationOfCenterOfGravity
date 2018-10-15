import numpy as np
import cv2
import os
from collections import deque

# firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

#Pyrebase
import pyrebase

#dotenv
import settings
JSON_PATH = settings.JP
API_KEY = settings.AP
AUTH_DOMAIN = settings.AD
DATABASE_URL = settings.DB
STORAGE_BUCKET = settings.SB

cred = credentials.Certificate(JSON_PATH)
firebase_admin.initialize_app(cred, {
    'databaseURL': DATABASE_URL,
    'databaseAuthVariableOverride': None
})

config = {
  'apiKey': API_KEY,
  'authDomain': AUTH_DOMAIN,
  'databaseURL': DATABASE_URL,
  'storageBucket': STORAGE_BUCKET
}
firebase = pyrebase.initialize_app(config)

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

if __name__ == '__main__':
    kernel = np.ones((5, 5), np.uint8)
    bpoints01 = gpoints01 = rpoints01 = ypoints01 = [deque(maxlen=512)]
    bindex01 = gindex01 = rindex01 = yindex01 = 0
    bpoints02 = gpoints02 = rpoints02 = ypoints02 = [deque(maxlen=512)]
    bindex02 = gindex02 = rindex02 = yindex02 = 0
    colors = [(0, 255, 0), (0, 0, 255)]

    fbUpperBodyTrajectory = []
    fbWaistTrajectory = []

    path = './video'
    file = os.listdir(path)
    fileSize = len(file)

    cap01 = cv2.VideoCapture(1)
    cap02 = cv2.VideoCapture(2)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out01 = cv2.VideoWriter('video/output' + str(1 + fileSize) +  '.webm',fourcc, 25, (1280, 960))
    out02 = cv2.VideoWriter('video/output' + str(2 + fileSize) + '.webm',fourcc, 25, (1280, 960))

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
            #cv2.circle(frame01, (int(rx + rw / 2), int(ry + rh / 2)), 5, (0,0,255), -1)
            center01 = (int(rx + rw / 2), int(ry + rh / 2))
            bpoints01[bindex01].appendleft(center01)
            fbUpperBodyTrajectory.append(center01)

            height = [0, 0]
            lx, ly, lw, lh = max(rectsGreen02, key=(lambda x: x[2] * x[3]))
            cv2.rectangle(frame02, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 3)
            #cv2.circle(frame02, (int(lx + lw / 2), int(ly + lh / 2)), 5, (0,0,255), -1)
            center02 = (int(lx + lw / 2), int(ly + lh / 2))
            bpoints02[bindex02].appendleft(center02)
            xBody.append(int(lx + lw / 2))
            xAbs += abs(xBody[l-1] - xBody[l])
            height = (xAbs, int(ly + lh / 2))
            fbWaistTrajectory.append(height)
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

        out01.write(frame01)
        out02.write(frame02)

        cv2.namedWindow("Upper body trajectory", cv2.WINDOW_NORMAL)
        cv2.imshow("Upper body trajectory", frame01)
        cv2.namedWindow("Waist trajectory", cv2.WINDOW_NORMAL)
        cv2.imshow("Waist trajectory", frame02)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap01.release()
            cap02.release()
            out01.release()
            out02.release()
            cv2.destroyAllWindows()
            break

        if key == ord("w"):
            bpoints01 = gpoints01 = rpoints01 = ypoints01 = [deque(maxlen=512)]
            bindex01 = gindex01 = rindex01 = yindex01 = 0
            bpoints02 = gpoints02 = rpoints02 = ypoints02 = [deque(maxlen=512)]
            bindex02 = gindex02 = rindex02 = yindex02 = 0

            fbUpperBodyTrajectory = []
            fbWaistTrajectory = []

            xBody = []
            xAbs = 0
            l = 0

            out01.release()
            out02.release()

            out01 = cv2.VideoWriter('video/output' + str(1 + fileSize) +  '.webm',fourcc, 25, (1280, 960))
            out02 = cv2.VideoWriter('video/output' + str(2 + fileSize) + '.webm',fourcc, 25, (1280, 960))

        if key == ord("s"):
            cv2.imwrite('photo/photo1.jpg', frame01)
            cv2.imwrite('photo/photo2.jpg', frame02)

    path01 = 'video/output' + str(1 + fileSize) + '.webm'
    path02 = 'video/output' + str(2 + fileSize) + '.webm'
    storage = firebase.storage()
    storage.child('video/output' + str(1 + fileSize) + '.webm').put(path01)
    storage.child('video/output' + str(2 + fileSize) + '.webm').put(path02)
    url01 = storage.child('video/output' + str(1 + fileSize) + '.webm').get_url(token=None)
    url02 = storage.child('video/output' + str(2 + fileSize) + '.webm').get_url(token=None)
    #print('svg01URL: {0} \nsvg02URL: {1}'.format(url02, url01))

    ref = db.reference('/public_resource')
    ref.push({
        'UpperBodyTrajectory': fbUpperBodyTrajectory,
        'WaistTrajectory': fbWaistTrajectory,
        'URL01': url01,
        'URL02': url02
    })

    print(ref.get())
