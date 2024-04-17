import os
import cv2
import json
import random
import re
import numpy as np
from PIL import Image
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


# GOALS:
# 1. Get coord_Data_Appointing to work first
#   - Once working, data_splitting should be complete
# 2. Get the dataAdjusting to work
#   - Still looking into this
# 3. Leave testing for last


# Groups coordinates of Waldo with the image
# STATUS: INCOMPLETE
def coord_Data_Appointing(dataset, coord_file, w):
    # Opens the coordinates file
    with open(coord_file, 'r') as f:
        coords = json.load(f)

    imgData_pairs = []

    # Retrieves the image name and the coordinates
    # Appends them together and preps them for coord appointing
    for img in dataset:
        imgPath = img[0]
        imgName = os.path.basename(imgPath)
        print(f"imgName: {imgName}")
        pattern = r'\d+'
        digits = re.findall(pattern, imgName)
        if len(digits[0]) == 1:
            digits[0] = '0' + digits[0]
        imgData_pairs.append((imgName, digits)) 

    imgData_pairs = sorted(imgData_pairs, key=lambda x: x[1])
    print(dataset[0][0])

    # Coordinates are assigned to the image
    for img in imgData_pairs:
        imgName = img[0]



# Splits the data into training, validation, and testing sets
# STATUS: INCOMPLETE
def data_splitting(dir1, dir2, w):
    # Each directory (waldo, not waldo) are given a binary classification
    sec1 = [(os.path.join(dir1, f), 0) for f in os.listdir(dir1)]
    sec2 = [(os.path.join(dir2, f), 1) for f in os.listdir(dir2)]

    coordsFile = os.path.join('Hey-Waldo-master', 'data.json')
    coord_Data_Appointing(sec1, coordsFile, w)
    
    # Combines both sets and splits them into training, validation, and testing sets
    # 70% training, 20% validation, 10% testing
    dataset = sec1 + sec2
    train, test = train_test_split(dataset, test_size=0.3, random_state=42)
    valid, test  =train_test_split(test, test_size=0.33, random_state=42)
    return train, valid, test


# Adjusts the data to the appropriate size
# STATUS: INCOMPLETE
def dataAdjusting(imgDataset, coordinates_File):
    with open(coordinates_File, 'r') as f:
        coordinates = f.load(f)
    imgs = []
    labels = []
    coords = []
    for imgPath, l in imgDataset:
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (width, height))
        #img = img / 255.0
        imgs.append(img)
        #img_coords = coords[width][i + 1][]

    imgs = np.array(imgs)
    return imgs, labels

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
not_waldo_dir = os.path.join('Hey-Waldo-master', '64', 'notwaldo')

# Current step is in the works
trainingSet, validSet, testingSet = data_splitting(waldo_dir, not_waldo_dir, 64)

print(f"Training Set: {len(trainingSet)}")
print(f"Validation Set: {len(validSet)}")
print(f"Testing Set: {len(testingSet)}")


#######################################################################
# Gets the image size parameters
# STATUS: INCOMPLETE
imgPath = os.path.join('Hey-Waldo-master', '64', 'waldo', '1_1_1.jpg')
imgPath = testingSet[0][0]
print(f"imgPath: {imgPath}")
img = Image.open(imgPath)
width = img.width
height = img.height
imageSize = (width, height)
print(f"Image Size: {imageSize}")

xTrain, yTrain = dataAdjusting(trainingSet)
xValid, yValid = dataAdjusting(validSet)

#######################################################################
# CNN Model based from NVIDIA Code with modificiations
# STATUS: DONE

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='linear'))  # Two output neurons for the x and y coordinates

model.compile(optimizer='adam', loss='mean_squared_error')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain , validation_data=(xValid, yValid), 
          epochs=10, batch_size=4)
#######################################################################

#######################################################################
# TESTING
# STATUS: INCOMPLETE
for items in range(len(testingSet)):
    getImg = testingSet[items][0]
    print(f"===\n i = {items} \n===")
    print(f"Image: {getImg} and Number: {testingSet[items][1]}")
    imgResults = predictions(getImg, width, height, model)
    print(f"Results: {imgResults}")

print("\n\n=== === ===\nNOW TESTING WITH WALDO")
imgWPATH = os.path.join('Hey-Waldo-master', '64', 'waldo', '19_0_7.jpg')
imgWALDORes = predictions(imgWPATH, width, height, model)
print(f"Waldo Results: {imgWALDORes}")

