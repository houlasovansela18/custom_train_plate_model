import cv2

import numpy as np



# capturing video

cap = cv2.VideoCapture('vid_03.mp4')


def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts,boundingBoxes

# reading back-to-back frames(images) from video

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

       # Difference between frame1(image) and frame2(image)
        diff = cv2.absdiff(frame1, frame2)

       # Converting color image to gray_scale image
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

       # Converting gray scale image to GaussianBlur, so that change can be find easily 
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

       # If pixel value is greater than 20, it is assigned white(255) otherwise black
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=4)

       # finding contours of moving object
        contours, hirarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

       # making rectangle around moving object

            
        try:
            conts,boundingBox  = sort_contours(contours)
            sorted_by_second = sorted(boundingBox, key=lambda tup: tup[3],reverse=True)
            (x, y, w, h) = sorted_by_second[0]
            if w >= 150 and h >= 150:
                print(w,h)
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255,0), 1)
                car_crop = frame1[y:y+h,x:x+w]
                cv2.imshow('Car Crop', car_crop)
        except:
            pass

       # Display original frame
        cv2.imshow('Motion Detector', frame1)

       # Display Diffrenciate Frame
    #    cv2.imshow('Difference Frame', thresh)

       # Assign frame2(image) to frame1(image)
        frame1 = frame2

       #Read new frame2
        ret, frame2 = cap.read()

       # Press 'esc' for quit 
        if cv2.waitKey(40) == 27:
                break

# Release cap resource
cap.release()

# Destroy all windows
cv2.destroyAllWindows()