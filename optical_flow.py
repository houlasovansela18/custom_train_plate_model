import numpy as np
import cv2 as cv

cap = cv.VideoCapture('vid_02.mp4')
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()


while(1):
    x_col, y_col = [],[]
    x_min, x_max, y_min, y_max,k_value = 0,0,0,0,0
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
    cont, _  = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cont:
        (x, y, w, h) = cv.boundingRect(c)
        if w/h > 1.3 and w>=100:
            x_col.append((x,w))
            y_col.append((y,h))
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0,255), 2)
    try:
        x_min = min(x_col)[0]
        y_min = min(y_col)[0]
        x_max = max([ i[0]+i[1]  for i in x_col])
        y_max = max([ i[0]+i[1]  for i in y_col])
        k_value = int(x_max - (x_max-x_min)*0.2)

    except:pass
    cv.rectangle(frame, (x_min, y_min), (k_value, y_max), (0, 255,0), 2)
    # car_crop = frame[k_value:y_max, x_min:x_max]
    cv.imshow('frame',frame)
