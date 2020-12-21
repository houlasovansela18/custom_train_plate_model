import cv2
import numpy as np
import keras
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import queue
import threading
import time
import re

from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder


def load_model(path):

    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print("Error",e)

def run_load_model():

    # load plate detection model>>
    
    wpod_net_path = "wpod-net.json"
    wpod_net = load_model(wpod_net_path)

    # Load model architecture, weight and labels
    json_file = open('MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("License_character_recognition_weight.h5")
    print("[INFO] Model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load('license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")

    return wpod_net,model,labels

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    # img = image_path
    # img = cv2.convertScaleAbs(img, alpha=0.5, beta=100)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path, Dmax=608, Dmin=440):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, lp_type, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.85)
    return vehicle, LpImg,lp_type, cor

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts,boundingBoxes

def prep_image(image_path):

    # Obtain plate image and its coordinates from an image
    test_image = image_path
    _, LpImg,lp_type,_ = get_plate(test_image)
    print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])

    if (len(LpImg)):
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        if lp_type == 1: plate_image = plate_image[15:plate_image.shape[0] - 17, 10:plate_image.shape[1]-15]
        else:plate_image = plate_image[10:plate_image.shape[0] - 30, 0:plate_image.shape[1]-10]
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)    
        # Applied inversed thresh_binary 
        binary = cv2.threshold(blur, 0, 255,cv2.THRESH_BINARY_INV +  cv2.THRESH_OTSU)[1]
        # check to find contour more better for sementation.

        cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cont,binary,plate_image,lp_type
    
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def drawKhmer_cont(test_roi,x_col,y_col,lp_type):

    try :
        x_min = min(x_col)[0]
        y_min = min(y_col)[0]
        x_max = max([ i[0]+i[1]  for i in x_col])
        y_max = max([ i[0]+i[1]  for i in y_col])
        
        if lp_type == 2:
            # cv2.rectangle(test_roi, (x_min, y_min), (x_max, y_max+10), (0, 255,0), 1)
            khmer_org_crop = test_roi[y_min:y_max+10, x_min:x_max]
        if lp_type == 1: 
            # cv2.rectangle(test_roi, (x_min, y_min), (x_max, y_max), (0, 255,0), 1)
            khmer_org_crop = test_roi[y_min:y_max, x_min:x_max]
        
        return khmer_org_crop
        
    except: pass

def detection_char(cont,binary,plate_image,lp_type):

    # create a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # define standard width and height of character
    digit_w, digit_h = 30, 60
    x_col, y_col = [],[]
    cont, _ = sort_contours(cont)

    crop_characters = []
    for c in cont:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 0,0), 1)
        if lp_type == 2:
            if 0.22<=x/plate_image.shape[1]<=0.73 and 0<=y/plate_image.shape[0]<=0.32:
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 0,255), 1)
                x_col.append((x,w))
                y_col.append((y,h))
            ratio = h/w
            if 1<=ratio<=6: # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.32 : # Select contour which has the height larger than 35% of the plate
                    # Draw bounding box around digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 1)
                    # Seperrate number and gibe prediction
                    curr_num = binary[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
        if lp_type == 1:
            if 0<=x/plate_image.shape[1] <=0.28 and 0.1<= y/plate_image.shape[0]<=0.7: 
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 0,255), 1)
                x_col.append((x,w))
                y_col.append((y,h))
            ratio = h/w
            if 1<=ratio<=6:  # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.32 and x/plate_image.shape[1]>=0.28: # Select contour which has the height larger than 35% of the plate
                    # Draw bounding box around digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 1)
                    # Seperrate number and gibe prediction
                    curr_num = binary[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)
    
    khmer_org_crop = drawKhmer_cont(test_roi,x_col,y_col,lp_type)

    print("Detect {} letters...".format(len(crop_characters)))
    # fig = plt.figure(figsize=(10,6))
    # plt.imshow(test_roi)
    # plt.show()
    return crop_characters,khmer_org_crop

def recognition_char(crop_characters):


    # fig = plt.figure(figsize=(15,3))
    # cols = len(crop_characters)
    # grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
    final_string = ''
    for i,character in enumerate(crop_characters):
        # fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character,model,labels))
        # if title.strip("'[]") == "P":
        #     cv2.imwrite("dataset_characters/R/R_1017.jpg",character)
        # plt.title('{}'.format(title.strip("'[]"),fontsize=20))
        final_string+=title.strip("'[]")
        # plt.axis(False)
        # plt.imshow(character,cmap='Blues_r')

    # plt.show()
    return final_string


def Receive():
    print("start Receive")
#     cap = cv2.VideoCapture("rtsp://admin:admin@10.2.7.251:554/1")
    cap = cv2.VideoCapture("vid_04.mp4")
    ret, frame = cap.read()
    # height, width, channels = frame.shape

#     q.put(frame[30:height, 0: width // 2])
    q.put(frame)
#    while ret:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            height, width, channels = frame.shape
            q.put(frame)
#             q.put(frame[30:height, 0: width // 2])
            time.sleep(0.3)

def Display():
    print("Start Displaying")
    time.sleep(1)
    while True:
        frame = q.get()
        cv2.imwrite("frame1.png", frame)
        print(frame.shape)

        # for testing 
        try :
            cont,binary,plate_image,lp_type = prep_image("frame1.png")

            # Initialize a list which will be used to append charater images
            crop_characters,khmer_org_crop = detection_char(cont,binary,plate_image,lp_type)
            if len(crop_characters) >= 5: 
                plate_id = recognition_char(crop_characters)
                print(plate_id)
            else: print("No plate ID detected>>")
        except: pass

if __name__ == "__main__":

    wpod_net,model,labels  = run_load_model()
    q = queue.Queue()
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()

    # cont,binary,plate_image,lp_type = prep_image("Plate_examples/khmer_35_car.jpg")
    # # cont,binary,plate_image,lp_type = prep_image("frame1.png")

    # # Initialize a list which will be used to append charater images
    # crop_characters = detection_char(cont,binary,plate_image)

    # result = recognition_char(crop_characters)
    # print(result)