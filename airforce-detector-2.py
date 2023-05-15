import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.resnet import ResNet50
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

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

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_files, target_size=(224, 224), batch_size=32)
test_data = test_datagen.flow_from_directory(test_files, target_size=(224, 224), batch_size=32)
numberOfAirplaneType = 40

output_shape = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).output_shape[1:]

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(units=numberOfAirplaneType, activation='softmax'))

for layer in base_model.layers[:-5]:
    layer.trainable = False

optimizer = Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
print(model.summary())

batch_size = 32

train_samples = count_samples(train_files)
test_samples = count_samples(test_files)

steps_per_epoch = train_samples // batch_size
validation_steps = test_samples // batch_size

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_data,
          steps_per_epoch=steps_per_epoch,
          epochs=40,
          validation_data=test_data,
          validation_steps=validation_steps,
          callbacks=[early_stopping])

model.save("plane_detector")