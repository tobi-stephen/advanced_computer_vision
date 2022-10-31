import main
import cv2 as cv
import time
import math
import numpy as np
import autopy

wc, hc = 640, 480
tipIds = [4, 8, 12, 16, 20]

def hmain():
    cap = cv.VideoCapture(0)
    cap.set(3, wc)
    cap.set(4, hc)
    p_time = 0
    hd = main.HandDetector(detConf=0.7)

    while True:
        ret, img = cap.read()

        img = hd.findHands(img)
        lm_list, bbox = hd.findPosition(img)
        # print(bbox)
        fingers = [0, 0, 0, 0, 0]
        if lm_list:
            # print(lm_list[4], lm_list[8])
            x1, y1 = lm_list[12][1:]
            x2, y2 = lm_list[8][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            #for thumb
            if lm_list[tipIds[0]][1] < lm_list[tipIds[0]-1][1]:
                fingers[0] = 1
            else:
                fingers[0] = 0

            for x, i in enumerate(tipIds[1:], 1):
                if lm_list[i][2] < lm_list[i-2][2]:
                    fingers[x] = 1
                else:
                    fingers[x] = 0

            print(fingers.count(1))

            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            # cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
            #
            # ll = math.hypot(x2 - x1, y2 - y1)
            # print(ll)
            #
            # if ll < 35:
            #     cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv.putText(img, f'FPS: {int(fps)}', (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
        cv.imshow('img', img)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    hmain()
