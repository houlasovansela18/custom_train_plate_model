import cv2
import numpy as np
import keras
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
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
    _ , LpImg, lp_type, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg,lp_type, cor

# Obtain plate image and its coordinates from an image
test_image = "Plate_examples/khmer_29_car.png"
vehicle, LpImg,lp_type,cor = get_plate(test_image)
print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])


if len(LpImg): #check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)



def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts,boundingBoxes

cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# create a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 30, 60
x_col, y_col = [],[]
cont, boundingBox = sort_contours(cont)
x_min, x_max, y_min, y_max = 0,0,0,0

for c in cont:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 0,0), 1)
    if lp_type == 1:
        ratio = h/w
        if 1<=ratio<=6: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.32  and x/plate_image.shape[1]>=0.32: # Select contour which has the height larger than 35% of the plate
                    # Draw bounding box around digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 1)
                x_col.append(x)
                x_col.append(x+w)
                y_col.append(y)
                y_col.append(y+h)
                    # Seperrate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
    if lp_type == 2:
        ratio = h/w
        if 1<=ratio<=6 and w <= 40: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.32: # Select contour which has the height larger than 35% of the plate
                    # Draw bounding box around digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 1)
                x_col.append(x)
                x_col.append(x+w)
                y_col.append(y)
                y_col.append(y+h)
                    # Seperrate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

if lp_type == 2: 

    try:
        # crop_khmer = plate_image[5:min(y_col),65:test_roi.shape[1]-65]
        cv2.rectangle(test_roi, (65, 5), (test_roi.shape[1]-65, min(y_col)-5), (0, 255,0), 1)
    except:pass
    

if lp_type == 1: 

    try:
        # crop_khmer = plate_image[5:min(y_col),65:test_roi.shape[1]-65]
        cv2.rectangle(test_roi, (20, 20), (min(x_col)-30,test_roi.shape[0]-40), (0, 255,0), 1)
    except:pass

print("Detect {} letters...".format(len(crop_characters)))
fig = plt.figure(figsize=(10,6))
plt.axis(False)
plt.imshow(test_roi)
plt.show()

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

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

fig = plt.figure(figsize=(15,3))
cols = len(crop_characters)
grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
final_string = ''
for i,character in enumerate(crop_characters):
    fig.add_subplot(grid[i])
    title = np.array2string(predict_from_model(character,model,labels))
    # if title.strip("'[]") == "3":
    #     cv2.imwrite("dataset_characters/2/2_1017.jpg",character)
    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
    final_string+=title.strip("'[]")
    plt.axis(False)
    plt.imshow(character,cmap='Blues_r')

print(final_string)
plt.show()
