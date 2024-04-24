import os
import cv2
import json
import random
import re
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

from tensorflow.keras.layers import (
    Dense, Conv2D, MaxPool2D, 
    Flatten, Dropout, BatchNormalization,
)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split
from keras import backend as K


# GOALS:
# 1. Get coord_Data_Appointing to work first
#   - Once working, data_splitting should be complete
# 2. Get the dataAdjusting to work
#   - Still looking into this
# 3. Leave testing for last


# Groups coordinates of Waldo with the image
# STATUS: COMPLETE?
def coord_Data_Appointing(dataset, coord_file, w, binary, hasFile):
    # Opens the coordinates file
    if(hasFile == True):
        with open(coord_file, 'r') as f:
            coords = json.load(f)
    imgData_pairs = []
    imgDirect = os.path.dirname(dataset[0][0])
    # Retrieves the image name and the coordinates
    # Appends them together and preps them for coord appointing
    for img in dataset:
        imgPath = img[0]
        imgName = os.path.basename(imgPath)
        if(imgName != '.DS_Store'):
            pattern = r'\d+'
            digits = re.findall(pattern, imgName)
            if len(digits[0]) == 1:
                digits[0] = '0' + digits[0]
            imgData_pairs.append((imgName, digits)) 

    imgData_pairs = sorted(imgData_pairs, key=lambda x: x[1])

    # Coordinates are assigned to the image
    for img in imgData_pairs:
        imgName = img[0]
        # Get coords
        xPos = -w
        yPos = -w
        if hasFile == True:
            # issue?
            for item in coords[str(int(w))][str(int(img[1][0]))]:
                if(item["x"] == (img[1][1]) and item["y"] == (img[1][2])):
                    xPos = item["x_px"] / 64
                    yPos = item["y_px"] / 64
                    break

        xyCoords = [xPos, yPos]
        dataIndx = dataset.index((imgDirect + '/' + imgName, binary))
        
        temp = list(dataset[dataIndx])
        temp.append(xyCoords)
        dataset[dataIndx] = tuple(temp)        

# Splits the data into training, validation, and testing sets
# STATUS: COMPLETE?
def data_splitting(dir1, dir1_1, dir1_2, dir2, w):
    # Each directory (waldo, not waldo) are given a binary classification
    sec1 = [(os.path.join(dir1, f), 0) for f in os.listdir(dir1)]
    sec1_1 = [(os.path.join(dir1_1, f), 0) for f in os.listdir(dir1_1)]
    sec1_2 = [(os.path.join(dir1_2, f), 0) for f in os.listdir(dir1_2)]
    sec2 = [(os.path.join(dir2, f), 1) for f in os.listdir(dir2)]

    coordsFile = os.path.join('Hey-Waldo-master', 'data.json')
    print(type(sec1))
    coord_Data_Appointing(sec1, coordsFile, w, 0, True)
    # =============
    # EXPERIMENTAL
    coord_Data_Appointing(sec1_1, coordsFile, w, 0, True)
    coord_Data_Appointing(sec1_2, coordsFile, w, 0, True)
    # EXPERIMENTAL
    # =============
    coord_Data_Appointing(sec2, coordsFile, w, 1, False)
    
    # Combines both sets and splits them into training, validation, and testing sets
    # 70% training, 20% validation, 10% testing
    #dataset = sec1 + sec1_1 + sec1_2 + sec2
    container = []
    for j in range(41):
        container = container + sec1 + sec1_1 + sec1_2

    dataset = container + sec2
    
    #dataset = sec1 + sec2
    train, test = train_test_split(dataset, test_size=0.3, random_state=42)
    valid, test = train_test_split(test, test_size=0.33, random_state=42)
    return train, valid, test


# Adjusts the data to the appropriate size
# STATUS: INCOMPLETE
def dataAdjusting(imgDataset, width, height):
    imgs = []
    coords = []
    for imgPath in imgDataset:
        if(os.path.basename(imgPath[0]) != '.DS_Store'):
            img = cv2.imread(imgPath[0])
            img = cv2.resize(img, (width, height))
            img = img / 255.0
            imgs.append(img)

            normCoords = [(imgPath[2][0])/width, (imgPath[2][1])/height]
            coords.append(normCoords)

    imgs = np.array(imgs)
    #coords = (np.array(coords))/(width * height)
    coords = (np.array(coords))

    return imgs, coords

def predictions(imgPath, width, height, model):
    image = image_utils.load_img(imgPath, target_size=(width, height))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,width,height,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds


#######################################################################
#######################################################################
# START OF PROGRAM
#######################################################################
#######################################################################

# Combining the data and splitting into the appropriate sizes
# STATUS: INCOMPLETE
waldo_dir = os.path.join('Hey-Waldo-master', '64', 'waldo')
waldo_bw_dir = os.path.join('Hey-Waldo-master', '64-bw', 'waldo')
waldo_grey_dir = os.path.join('Hey-Waldo-master', '64-gray', 'waldo')
not_waldo_dir = os.path.join('Hey-Waldo-master', '64-copy', 'notwaldo')

# Current step is in the works
trainingSet, validSet, testingSet = data_splitting(waldo_dir, waldo_bw_dir, waldo_grey_dir, not_waldo_dir, 64)

print(f"Training Set: {len(trainingSet)}")
print(f"Validation Set: {len(validSet)}")
print(f"Testing Set: {len(testingSet)}")


#######################################################################
# Gets the image size parameters
# STATUS: INCOMPLETE
imgPath = os.path.join('Hey-Waldo-master', '64', 'waldo', '1_1_1.jpg')
imgPath = testingSet[0][0]

img = Image.open(imgPath)
width = img.width
height = img.height
imageSize = (width, height)

xTrain, yTrain = dataAdjusting(trainingSet, width, height)
xValid, yValid = dataAdjusting(validSet, width, height)

imgGen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
imgGen.fit(xTrain)

#######################################################################
# CNN Model based from NVIDIA Code with modificiations
# STATUS: DONE?

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='linear'))  # Two output neurons for the x and y coordinates

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xTrain, yTrain , validation_data=(xValid, yValid), 
          epochs=10, batch_size=4)

# ADDING A NEW MODEL
base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(width, height, 3),
    include_top=False
)
base_model.trainable = False

inputs = keras.Input(shape=(width, height, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(2)(x)
vgg16_model = keras.Model(inputs, outputs)

vgg16_model.compile(optimizer='adam', loss='mean_squared_error')
vgg16_model.fit(xTrain, yTrain , validation_data=(xValid, yValid), 
          epochs=10, batch_size=4)




#model.fit(imgGen.flow(xTrain, yTrain, batch_size=4), validation_data=(xValid, yValid),
#            epochs=10)
#######################################################################
#######################################################################
# TESTING
# FOR DISPLAY PURPOSES
#for items in range(len(testingSet)):
#    getImg = testingSet[items][0]
#    print(f"===\n i = {items} \n===")
#    print(f"Image: {getImg} and Number: {testingSet[items][1]}")
#    imgResults = predictions(getImg, width, height, model)
#    print(f"Results: {imgResults}")

# TESTING WITH TWO MODELS

print("\n\n=== === ===\nNOW TESTING WITH WALDO - Scratch Model")
imgWPATH = os.path.join('Hey-Waldo-master', '64', 'waldo', '18_15_7.jpg')
imgWALDORes = predictions(imgWPATH, width, height, model)
print(f"Waldo Results for {imgWPATH}: {imgWALDORes}")

imgWPATH = os.path.join('Hey-Waldo-master', '64', 'waldo', '11_6_11.jpg')
imgWALDORes = predictions(imgWPATH, width, height, model)
print(f"Waldo Results for {imgWPATH}: {imgWALDORes}")

imgWPATH = os.path.join('Hey-Waldo-master', '64', 'waldo', '19_0_7.jpg')
imgWALDORes = predictions(imgWPATH, width, height, model)
print(f"Waldo Results for {imgWPATH}: {imgWALDORes}")


print("\n\n=== === ===\nNOW TESTING WITH WALDO - Prebuilt Model")
imgWPATH = os.path.join('Hey-Waldo-master', '64', 'waldo', '18_15_7.jpg')
imgWALDORes = predictions(imgWPATH, width, height, vgg16_model)
print(f"Waldo Results for {imgWPATH}: {imgWALDORes}")

imgWPATH = os.path.join('Hey-Waldo-master', '64', 'waldo', '11_6_11.jpg')
imgWALDORes = predictions(imgWPATH, width, height, vgg16_model)
print(f"Waldo Results for {imgWPATH}: {imgWALDORes}")

imgWPATH = os.path.join('Hey-Waldo-master', '64', 'waldo', '19_0_7.jpg')
imgWALDORes = predictions(imgWPATH, width, height, vgg16_model)
print(f"Waldo Results for {imgWPATH}: {imgWALDORes}")
# Freeing memory just in case
del model
K.clear_session()
