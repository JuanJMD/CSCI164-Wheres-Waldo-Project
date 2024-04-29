import os
import cv2
import random
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


def data_splitting(dir1, dir2):
    sec1 = [(os.path.join(dir1, f), 0) for f in os.listdir(dir1)]
    sec2 = [(os.path.join(dir2, f), 1) for f in os.listdir(dir2)]

    dataset = sec1 + sec2

    train, test = train_test_split(dataset, test_size=0.3, random_state=42)
    valid, test  =train_test_split(test, test_size=0.33, random_state=42)

    return train, valid, test

def dataAdjusting(imgDataset):
    imgs = []
    labels = []
    for imgPath, l in imgDataset:
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (width, height))
        #img = img / 255.0
        imgs.append(img)
        labels.append(l)
    
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels

def predictions(imgPath, width, height, model):
    image = image_utils.load_img(imgPath, target_size=(width, height))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,width,height,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds

# Combining the data and splitting into the appropriate sizes
waldo_dir = os.path.join('Hey-Waldo-master', '64', 'waldo')
not_waldo_dir = os.path.join('Hey-Waldo-master', '64', 'notwaldo')
trainingSet, validSet, testingSet = data_splitting(waldo_dir, not_waldo_dir)

print(f"Training Set: {len(trainingSet)}")
print(f"Validation Set: {len(validSet)}")
print(f"Testing Set: {len(testingSet)}")


# Gets the image size parameters
imgPath = os.path.join('Hey-Waldo-master', '256', 'waldo', '1_1_1.jpg')
imgPath = testingSet[0][0]
print(f"imgPath: {imgPath}")
img = Image.open(imgPath)
width = img.width
height = img.height
imageSize = (width, height)
print(f"Image Size: {imageSize}")

xTrain, yTrain = dataAdjusting(trainingSet)
xValid, yValid = dataAdjusting(validSet)
# CNN Model based from NVIDIA Code
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 3)))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xTrain, yTrain , validation_data=(xValid, yValid), 
          epochs=10, batch_size=4)


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

