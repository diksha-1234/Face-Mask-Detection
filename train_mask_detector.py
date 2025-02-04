# import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator#data augmentation
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize parameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
DIRECTORY = r"F:\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Load images
print("[INFO] loading images...")
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))#input size expected by mobilenetv2
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# Perform one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)#learn the parameters from input data(mean,variance) and apply transformation to data
labels = to_categorical(labels)

data = np.array(data, dtype="float32")#compatible with libraries like tensorflow/pytorch
labels = np.array(labels)

# Train-test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load MobileNetV2 and construct the head of the model
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Construct the final model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze layers in base model
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Make predictions and evaluate the model
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# Classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Save the model
print("[INFO] saving mask detector model...")
model.save("mask_detector.h5")

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()

# Ensure that keys exist before plotting
if "loss" in H.history and "val_loss" in H.history:
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

if "accuracy" in H.history and "val_accuracy" in H.history:
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

#include_top -allows us to add custom layers for classification tasks.
#pooling size reduce feature learned by mobilenetv2
#flatten 2-d to 1d
#add fully connected layer with 128 neurons help to learn complex patterns.
#to prevent overfitting randomly sets 50% of neurons to 0.
#.h5 is the file extension of hierarchial data format version 5(hdf5) binary file format to store large amount of data efficiently.
#freeze -if already pretrained for general tasks prevent it from accidently altered or degraded during training.
#Mobilenetv2 -cnn model-already pretrained on imagenet.
#preprocessing->custom model using mobilenetv2->training of dataset on model->evaluation of model
