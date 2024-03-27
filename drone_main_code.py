from drone_basic import *
from function import *
from hand_detect import *
import cv2
from time import sleep

w,h=360,240
drone=drone_basic()

pid=[0.5, 0.5, 0]
pError=0
start=0

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


colors = [(245,117,16), (117,245,16), (16,117,245)]

sequence = []
sentence = []
threshold = 0.2

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:

        if start == 0:
            drone.takeoff()
            drone.send_rc_control(0,0,10,0)
            sleep(2)
            start=1

        model = load_weights()

        img=camera(drone,w,h)

        img, info=Find_Face(img)
        print(info[0][0])

        pError=Track_Face(drone,info,w,pid,pError)

        b,img=detect_hand(img,holistic)

        if b=="iloveyou":
            drone.flip_back()
        elif b=="thank":
            drone.send_rc_control(0, 0, 10, 0)
            sleep(2)
            drone.send_rc_control(0, 0, -10, 0)
            sleep(2)
        elif b=="hello":
            drone.send_rc_control(10, 0, 0, 0)
            sleep(2)
            drone.send_rc_control(-10, 0, 0, 0)
            sleep(2)


        cv2.imshow('image',img)

        if cv2.waitKey(1) and 0xff==ord('1'):
            drone.land()
            break