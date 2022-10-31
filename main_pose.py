import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mp_drawing = mp.solutions.drawing_utils


def main():
    p_time = 0
    scale = 0.7
    cap = cv.VideoCapture('vid/darina.mp4')
    # cap = cv.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            print('no frame')
            break

        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = pose.process(rgb)
        print(res.pose_landmarks)
        img = cv.resize(img, None, img, scale, scale)

        if res.pose_landmarks:
            mp_drawing.draw_landmarks(img, res.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(mp_drawing.RED_COLOR))
            for id, lm in enumerate(res.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # if draw:
                cv.circle(img, (cx, cy), 10, (100, 200, 120), 4)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        img = cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)

        cv.imshow('img', img)
        if cv.waitKey(30) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
