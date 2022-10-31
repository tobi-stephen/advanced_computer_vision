import enum
import numpy as np
import cv2 as cv
import mediapipe as mp
import time
import sys


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detConf=0.5, trackConf=0.5) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detConf = detConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detConf, self.trackConf)
        self.mp_drawing = mp.solutions.drawing_utils

        self.res = None

    def findHands(self, img, draw=True):
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.res = self.hands.process(rgb)

        if self.res.multi_hand_landmarks:
            # print(len(res.multi_hand_landmarks))
            for hlm in self.res.multi_hand_landmarks:
                # for id, lm in enumerate(hlm.landmark):
                #     h, w, c = img.shape
                #     cx, cy = int(lm.x*w), int(lm.y*h)

                if draw:
                    self.mp_drawing.draw_landmarks(img, hlm, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        xmin = sys.maxsize
        ymin = sys.maxsize
        xmax = -sys.maxsize
        ymax = -sys.maxsize

        if self.res and self.res.multi_hand_landmarks:
            hand = self.res.multi_hand_landmarks[handNo]
            for i, lm in enumerate(hand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xmin = min(xmin, cx)
                xmax = max(xmax, cx)
                ymin = min(ymin, cy)
                ymax = max(ymax, cy)

                if draw:
                    cv.circle(img, (cx, cy), 10, (100, 200, 120), 4)

                lmList.append((i, cx, cy))

            if draw:
                cv.rectangle(img, (xmin-20, ymin-20), (xmax+20,ymax+20), (100, 200, 120), 4)

        return lmList, (xmin,ymin, xmax,ymax)


def main():
    p_time = 0
    cap = cv.VideoCapture(0)
    hd = HandDetector()

    while True:
        ret, img = cap.read()

        img = hd.findHands(img)
        lmList = hd.findPosition(img)
        if lmList:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - p_time)
        p_time = cTime
        img = cv.putText(cv.flip(img, 1), str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)

        cv.imshow('img', img)
        k = cv.waitKey(5)
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
