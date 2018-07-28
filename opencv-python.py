import numpy as np
import cv2

# Red, Blue, GreenのHSVの範囲を定義
# class Red:
#     def __init__(self):
#         self.lower = np.array([150, 50, 50])
#         self.upper = np.array([180, 255, 255])
#
# class Blue:
#     def __init__(self):
#         self.lower = np.array([110, 50, 50])
#         self.upper = np.array([130, 255, 255])

class Green:
    def __init__(self):
        self.lower = np.array([40, 50, 50])
        self.upper = np.array([80, 255, 255])

def colorTracking(frame, colorObj):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    mask = np.zeros(h.shape, dtype=np.uint8)
    mask = cv2.inRange(hsv, colorObj.lower, colorObj.upper)

    image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    for contour in contours:
        convexHull = cv2.convexHull(contour)
        boundingRect = cv2.boundingRect(convexHull)
        rects.append(np.array(boundingRect))

    return rects


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    fontType = cv2.FONT_HERSHEY_SIMPLEX

    while(True):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)
            rectsGreen = colorTracking(frame, Green())

            if len(rectsGreen) > 0:
                rect = max(rectsGreen, key=(lambda x: x[2] * x[3]))
                x, y, w, h = rect
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
                cv2.circle(frame, (int(x+w/2), int(y+h/2)), 5, (0,0,255), -1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
