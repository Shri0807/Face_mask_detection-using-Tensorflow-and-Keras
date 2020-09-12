import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
##############################

DIRECTORY = r'E:\Projects\Facemask\dataset'
CATEGORIES = ["with_mask", "without_mask"]

INIT_LR = 0.0001
EPOCHS = 20
BS = 32
###############################

data = []
labels = []
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

########## PERFORM ONE HOT ENCODING
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype='float32')
labels = np.array(labels)

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2,
                                                      stratify=labels, random_state=42)

############### IMAGE AUGMENTATION
ag = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest')

############### LOAD MobileNetV2 (No top layers)
base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

############### CONSTRUCT MODELS ON TOP OF base_model
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

############ PLACE CONSTRUCTED MODEL ON TOP OF BASE MODEL
model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable = False

########## COMPILE MODEL
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

########## TRAIN MODEL
print('STARTING TRAIN PROCESS.....')
history = model.fit(ag.flow(X_train, y_train, batch_size=BS),
                    steps_per_epoch=len(X_train) // BS,
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) // BS,
                    epochs=EPOCHS)

############ MAKE PREDICATIONS
pred = model.predict(X_test, batch_size=BS)

pred = np.argmax(pred, axis=1)

print(classification_report(y_test.argmax(axis=1),
                            pred,
                            target_names=lb.classes_))

model.save('mask_detection.model', save_format='h5')


############# PLOT GRAPH
N = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title('Training Loss and Accuracy')
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


