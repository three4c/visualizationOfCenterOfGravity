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

# def mouseEvent(event, x, y, flg, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print('START')
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         print('STOP')


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)
    fontType = cv2.FONT_HERSHEY_SIMPLEX

    while(True):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)

            # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            # cv2.setMouseCallback("frame", mouseEvent)

            rectsGreen = colorTracking(frame, Green())

            if len(rectsGreen) > 0:
                rx, ry, rw, rh = max(rectsGreen, key=(lambda x: x[2] * x[3]))
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
                cv2.circle(frame, (int(rx + rw / 2), int(ry + rh / 2)), 5, (0,0,255), -1)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
