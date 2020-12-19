import numpy as np
import cv2


cap = cv2.VideoCapture('vid_02.mp4')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


while(cap.isOpened()):
    x_col, y_col = [],[]
    x_min, x_max, y_min, y_max,k_value = 0,0,0,0,0
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cont, _  = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        (x, y, w, h) = cv2.boundingRect(c)
        if w/h > 1.3 and w>=100:
            x_col.append((x,w))
            y_col.append((y,h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0,255), 2)
    try:
        x_min = min(x_col)[0]
        y_min = min(y_col)[0]
        x_max = max([ i[0]+i[1]  for i in x_col])
        y_max = max([ i[0]+i[1]  for i in y_col])
        k_value = int(x_max - (x_max-x_min)*0.2)

    except:pass
    cv2.rectangle(frame, (x_min, y_min), (k_value, y_max), (0, 255,0), 2)
    car_crop = frame[k_value:y_max, x_min:x_max]
    # cv2.line(frame, (0,0), (1000,1000), (0, 255, 0) , 2) 
    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()