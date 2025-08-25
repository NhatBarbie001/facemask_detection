import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from data_utils import getJSON, adjust_gamma
from model import build_model

# ---- Dataset paths ----
directory = "../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/annotations"
image_directory = "../input/face-mask-detection-dataset/Medical mask/Medical mask/Medical Mask/images"

df = pd.read_csv("../input/face-mask-detection-dataset/train.csv")
df_test = pd.read_csv("../input/face-mask-detection-dataset/submission.csv")

# ---- Load JSON (labels) ----
jsonfiles = []
for i in os.listdir(directory):
    jsonfiles.append(getJSON(os.path.join(directory, i)))

# TODO: bạn cần xử lý jsonfiles để tạo ra biến data = [(image_array, label), ...]
# Ở đây mình giả định đã có data
# --------------------------
data = []  # <-- chỗ này bạn phải tự xử lý annotation thành (X, Y)

X = []
Y = []
for features, label in data:
    X.append(features)
    Y.append(label)

X = np.array(X)/255.0
X = X.reshape(-1,124,124,3)
Y = np.array(Y)

# ---- Build model ----
model = build_model(input_shape=(124,124,3))

# ---- Train/Validation split ----
xtrain, xval, ytrain, yval = train_test_split(X, Y, train_size=0.8, random_state=0)

# ---- Data augmentation ----
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)
datagen.fit(xtrain)

# ---- Training ----
history = model.fit(
    datagen.flow(xtrain, ytrain, batch_size=32),
    steps_per_epoch=xtrain.shape[0]//32,
    epochs=50,
    verbose=1,
    validation_data=(xval, yval)
)

# ---- Inference Demo ----
test_images = ['1114.png','1504.jpg', '0072.jpg','0012.jpg','0353.jpg','1374.jpg']
gamma = 2.0
assign = {'0':'Mask','1':"No Mask"}

fig = plt.figure(figsize=(14,14))
rows, cols = 3, 2
axes = []

# Load pretrained face detector
cvNet = cv2.dnn.readNetFromCaffe('weights.caffemodel')

for j, im in enumerate(test_images):
    image = cv2.imread(os.path.join(image_directory, im), 1)
    image = adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()

    for i in range(0, detections.shape[2]):
        try:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            confidence = detections[0, 0, i, 2]

            if confidence > 0.2:
                im_crop = cv2.resize(frame, (124,124))
                im_crop = np.array(im_crop)/255.0
                im_crop = im_crop.reshape(1,124,124,3)

                result = model.predict(im_crop)
                label_Y = 1 if result > 0.5 else 0

                cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
                cv2.putText(image, assign[str(label_Y)], (startX, startY-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)

        except:
            pass

    axes.append(fig.add_subplot(rows, cols, j+1))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.show()
