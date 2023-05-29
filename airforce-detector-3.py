import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import matplotlib.pyplot as plt

tf.device('/CPU:0'), tf.device('/GPU:1')
print(len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)
rescale_factor = 1./255
num_classes = 40
epochs = 20

train_data_dir = 'airplane-dataset-trans/train/'
validation_data_dir = 'airplane-dataset-trans/test/'

base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)


model = Model(inputs=base_model.input, outputs=predictions)


for layer in model.layers[:50]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define ImageDataGenerators without data augmentation but with rescaling
train_datagen = ImageDataGenerator(rescale=rescale_factor)
validation_datagen = ImageDataGenerator(rescale=rescale_factor)

# Define train and validation generators
train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(224, 224),
                                                              batch_size=32,
                                                              class_mode='categorical')


early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(train_generator,
          steps_per_epoch=None,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=None,
          callbacks=[early_stopping])


# Plot training & validation accuracy values
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Save the figure
plt.savefig('training_history.png')
plt.show()

model.save("plane_detector-3")