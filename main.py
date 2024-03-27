from drone_basic import *
import cv2

w,h=360,240
drone=drone_basic()

pid=[0.5, 0.5, 0]
pError=0
start=0

while True:

    if start == 0:
        drone.takeoff()
        start=1

    img=camera(drone,w,h)

    img, info=Find_Face(img)
    print(info[0][0])

    pError=Track_Face(drone,info,w,pid,pError)

    cv2.imshow('image',img)

    if cv2.waitKey(1) and 0xff==ord('1'):
        drone.land()
        break