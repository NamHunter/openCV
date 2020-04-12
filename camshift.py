import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import requests
import imutils
import dlib

url = "http://192.168.1.9:8080/shot.jpg?rnd=750019"

# cap = cv.VideoCapture('video\slow_traffic_small.mp4')
# cap = cv.VideoCapture(0)
# take first frame of the video
# ret, frame = cap.read()
# setup initial location of window


img_resp = requests.get(url)
img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
frame = cv.imdecode(img_arr,-1)

frame = cv.resize(frame,(400,300))
a = np.array(frame)
print(a.shape)

x, y, width, height = 0, 100, 300, 200
track_window = (x, y ,width, height)
# set up the ROI for tracking
roi = frame[y:y+height, x : x+width]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((146., 0., 106.)), np.array((181., 246., 255)))
roi_hist = cv.calcHist([hsv_roi], [0,1], mask, [256,256], [0, 256,0,256])



# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 2, 1)
cv.imshow('roi',roi)
while(1):
    # ret, frame = cap.read()

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
    frame = cv.imdecode(img_arr,-1)

    frame = cv.resize(frame,(400,300))


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0,1], roi_hist, [0, 256,0,256], 1)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    cv.filter2D(dst,-1,kernel)
    # apply meanshift to get the new location
    ret, track_window = cv.CamShift(dst, track_window, term_crit)

    # Draw it on image
    pts = cv.boxPoints(ret)
    print(pts)
    pts = np.int0(pts)
    final_image = cv.polylines(frame, [pts], True, (0, 255, 0), 2)
    #x,y,w,h = track_window
    #final_image = cv.rectangle(frame, (x,y), (x+w, y+h), 255, 3)

    cv.imshow('dst', dst)
    cv.imshow('final_image',final_image)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
