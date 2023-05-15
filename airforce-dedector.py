from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet import ResNet50
import matplotlib.pyplot as plt
import os

def count_samples(directory):
    count = 0
    for root, _, files in os.walk(directory):
        count += len(files)
    return count

train_files = "airplane-dataset-trans/augmented_train/"
test_files = "airplane-dataset-trans/test/"

img = load_img(test_files + "A-10_Thunderbolt/image (198).png")
print(img_to_array(img).shape)
plt.imshow(img)

train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(train_files, target_size=(224, 224), batch_size=32)
test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(test_files, target_size=(224, 224), batch_size=32)
numberOfAirplaneType = 40

output_shape = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).output_shape[1:]

model = Sequential()
model.add(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
model.add(GlobalAveragePooling2D())

for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=numberOfAirplaneType, activation='softmax'))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

batch_size = 32

train_samples = count_samples(train_files)
test_samples = count_samples(test_files)

steps_per_epoch = train_samples // batch_size
validation_steps = test_samples // batch_size

model.fit(train_data,
          steps_per_epoch=steps_per_epoch,
          epochs=10,
          validation_data=test_data,
          validation_steps=validation_steps)

model.save("planedetector4")