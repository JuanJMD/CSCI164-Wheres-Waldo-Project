import os
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


def make_predictions(image_path, w, h):
    image = image_utils.load_img(image_path, target_size=(w, h))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,w,h,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds

# Gets image size
imgPath = os.path.join('Hey-Waldo-master', '256', 'waldo', '1_1_1.jpg')
img = Image.open(imgPath)
width = img.width
height = img.height
imageSize = (width, height)

xTrain = os.path.join('Hey-Waldo-master', '256')
xValid = os.path.join('Hey-Waldo-master', 'original-images')

datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False, # Don't randomly flip images vertically
)

batch_train = 4
batch_valid = 4

train_gen = datagen.flow_from_directory(
    xTrain,
    target_size=imageSize,
    batch_size=batch_train,
    class_mode='binary'  # Binary classification (with Waldo or without Waldo)
)
valid_gen = datagen.flow_from_directory(
    xValid,
    target_size=imageSize,
    batch_size=batch_valid,
    class_mode='binary'

)
model = Sequential()
model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", 
                 input_shape=(width, height, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Dropout(0.3))  # Increase dropout rate
model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Dropout(0.3))  # Increase dropout rate
model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding="same"))
model.add(Flatten())
model.add(Dense(units=512, activation="relu", kernel_regularizer=regularizers.l2(0.01)))  # Add L2 regularization
model.add(Dropout(0.5))  # Increase dropout rate
model.add(Dense(units=1, activation="sigmoid"))  # Use 'sigmoid' for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print((len(train_gen))/batch_train)

model.fit(train_gen, steps_per_epoch = int((len(train_gen))/batch_train), epochs=10, 
          validation_data=valid_gen, validation_steps=len(valid_gen))

print('== PREDICTIONS ==\n')

result = make_predictions('Hey-Waldo-master/pepe.jpeg', width, height)
if result > 0.5:
    print("Waldo is present in the image!")
else:
    print("Waldo is not present in the image.")


result = make_predictions('Hey-Waldo-master/256/waldo/2_0_1.jpg', width, height)
if result > 0.5:
    print("Waldo is present in the image!")
else:
    print("Waldo is not present in the image.")

