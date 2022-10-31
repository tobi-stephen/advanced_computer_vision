import main
import cv2 as cv
import time
import math
import numpy as np
import mediapipe as mp

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wc, hc = 640, 480

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
print(volume.GetMasterVolumeLevel())
vol_range = volume.GetVolumeRange()
print(vol_range)
# volume.SetMasterVolumeLevel(-20, None)
min_vol = vol_range[0]
max_vol = vol_range[1]

def hmain():
    cap = cv.VideoCapture(0)
    cap.set(3, wc)
    cap.set(4, hc)
    p_time = 0
    hd = main.HandDetector(detConf=0.7)

    while True:
        ret, img = cap.read()

        img = hd.findHands(img)
        lm_list = hd.findPosition(img)
        if lm_list:
            # print(lm_list[4], lm_list[8])

            x1, y1 = lm_list[4][1:]
            x2, y2 = lm_list[8][1:]

            cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)

            ll = math.hypot(x2 - x1, y2 - y1)
            # print(ll)

            # hand range(for Spyke): 35 - 230
            # vol range: -96.0 - 0.0

            vol = np.interp(ll, [35, 220], [min_vol, max_vol])
            print(ll, " -> ", vol)
            volume.SetMasterVolumeLevel(vol, None)

            if ll < 35:
                cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

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
