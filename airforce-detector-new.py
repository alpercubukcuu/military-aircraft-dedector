import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import splitfolders
from keras.layers import Dense, Dropout, Flatten
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping


# Splitting the dataset
dataset_dir = 'airplane-dataset-trans/augmented_train/'
splitfolders.ratio(dataset_dir, output="splitted_dataset_v3", seed=42, ratio=(.8, .1, .1), group_prefix=None)

# Setting directories
train_dir = 'splitted_dataset_v3/train'
val_dir = 'splitted_dataset_v3/val'
test_dir = 'splitted_dataset_v3/test'

# Image dimensions
img_width, img_height = 224, 224

# ResNet model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freezing layers
for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer('conv5_block3_out')
print(last_layer.output_shape)

# Adding custom layers
x = Flatten()(last_layer.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(40, activation='softmax')(x)
model = Model(base_model.input, x)

# Compiling the model
opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

# Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(img_width, img_height),
                                                 color_mode='rgb',
                                                 batch_size=128,
                                                 class_mode='categorical')

validation_set = validation_datagen.flow_from_directory(val_dir,
                                                        target_size=(img_width, img_height),
                                                        color_mode='rgb',
                                                        batch_size=128,
                                                        class_mode='categorical',
                                                        shuffle=False)

# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

# Training the model
history = model.fit(training_set,
                              epochs=45,
                              validation_data=validation_set,
                              verbose=1,
                              callbacks=[es])

# Plotting the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(5, 5))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and val accuracy')

plt.figure(figsize=(5, 5))
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

# Evaluating the model
score = model.evaluate(validation_set)

# Testing the model
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(img_width, img_height),
                                            color_mode='rgb',
                                            batch_size=128,
                                            class_mode='categorical',
                                            shuffle=False)
score = model.evaluate(test_set)
model.save("plane_detector-new")