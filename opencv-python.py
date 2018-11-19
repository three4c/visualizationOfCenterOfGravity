import numpy as np
import cv2
import glob
import time
from collections import deque
from datetime import datetime

# Firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

#dotenv
import settings
JSON_PATH = settings.JP
DATABASE_URL = settings.DB

cred = credentials.Certificate(JSON_PATH)
firebase_admin.initialize_app(cred, {
    'databaseURL': DATABASE_URL,
    'databaseAuthVariableOverride': None
})

# Color Class
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

# Tracking Function
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
    print ('0: Expert\n1: Biginner01\n2: Biginner02\n3: Biginner03\n4: Biginner04\nPlease select one...')
    video = input('>>> ')
    print (('What was input... ') + video)

    if video == '0': resource = 'expert' # kanata
    elif video == '1': resource = 'biginner1' # どいちゃん
    elif video == '2': resource = 'biginner2' # わたる
    elif video == '3': resource = 'biginner3' # ジョージ
    else: resource = 'biginner4' # ボンちゃん

    kernel = np.ones((5, 5), np.uint8)
    bpoints = gpoints = rpoints = ypoints = [deque(maxlen=512)]
    bindex = gindex = rindex = yindex = 0
    colors = [(0, 255, 0), (0, 0, 255)]

    fbWaistTrajectory = []

    date = datetime.now().strftime('%Y.%m.%d')

    path = '../visualization/src/assets/video' + video + '/*.mp4'
    file = glob.glob(path)
    fileSize = len(file)

    cap = cv2.VideoCapture(1)
    fourcc = cv2.VideoWriter_fourcc(*'AVC1')
    out = cv2.VideoWriter('../visualization/src/assets/video' + video + '/output' + str(1 + fileSize) + '.mp4',fourcc, 20, (1280, 960))

    xBody = []
    xAbs = 0
    l = 0

    while(True):
        ret, frame = cap.read()
        if ret == True:
            rectsGreen = colorTracking(frame, Blue())

            if len(rectsGreen) > 0:
                height = [0, 0]
                lx, ly, lw, lh = max(rectsGreen, key=(lambda x: x[2] * x[3]))
                cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (0, 255, 0), 3)
                cv2.circle(frame, (int(lx + lw / 2), int(ly + lh / 2)), 5, (0,0,255), -1)
                center = (int(lx + lw / 2), int(ly + lh / 2))
                bpoints[bindex].appendleft(center)
                xBody.append(int(lx + lw / 2))
                xAbs += abs(xBody[l-1] - xBody[l])
                height = (xAbs, int(ly + lh / 2))
                fbWaistTrajectory.append(height)
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
                        # Line graph
                        front = points[i][j][k - 1]
                        back = points[i][j][k]
                        xBack += abs(front[0] - back[0])
                        cv2.line(frame, (xFront, front[1]), (xBack, back[1]), colors[1], 2)
                        xFront = xBack

            out.write(frame)

            cv2.namedWindow("Waist trajectory", cv2.WINDOW_NORMAL)
            cv2.imshow("Waist trajectory", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break

            if key == ord("w"):
                bpoints = gpoints = rpoints = ypoints = [deque(maxlen=512)]
                bindex = gindex = rindex = yindex = 0

                fbWaistTrajectory = []

                xBody = []
                xAbs = 0
                l = 0

                out.release()

                out = cv2.VideoWriter('../visualization/src/assets/video' + video + '/output' + str(1 + fileSize) + '.mp4',fourcc, 20, (1280, 960))
        else:
            time.sleep(2)

    fileName = 'output' + str(1 + fileSize) + '.mp4'

    # Firebase Realtime Database
    ref = db.reference('public_resource')
    childRef = ref.child(resource)
    childRef.push({
        'WaistTrajectory': fbWaistTrajectory,
        'Date': date,
        'FileName': fileName
    })

    print('Folder Name: video{0}\nLocal Foloder Size: {1}\nFile Name: {2}\nResource Name: {3}'.format(video, fileSize, fileName, resource))
    # print(ref.get())
