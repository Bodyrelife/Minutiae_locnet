import cv2
from math import pi, sin, cos

def show_mnt(image, mnt_list, length=20, color=(0, 255, 0)):
    I = cv2.imread(image)
    for x, y, o in mnt_list:
        center = (int(x), int(y))
        pointer = (int(x + length * cos(o)), int(y + length * sin(o)))
        cv2.circle(I, center, 4, color, 2)
        cv2.line(I, center, pointer, color, 2)
    cv2.imshow('Img', I)
    cv2.waitKey(0)
