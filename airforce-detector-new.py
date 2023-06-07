import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import splitfolders
from keras.layers import Dense, Dropout, Flatten
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping



dataset_dir = 'airplane-dataset-trans/augmented_train/'
splitfolders.ratio(dataset_dir, output="splitted_dataset_v3", seed=42, ratio=(.8, .1, .1), group_prefix=None)


train_dir = 'splitted_dataset_v3/train'
val_dir = 'splitted_dataset_v3/val'
test_dir = 'splitted_dataset_v3/test'


img_width, img_height = 224, 224

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))


for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer('conv5_block3_out')
print(last_layer.output_shape)


x = Flatten()(last_layer.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(40, activation='softmax')(x)
model = Model(base_model.input, x)


opt = tf.keras.optimizers.Adam(learning_rate=1e-6)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])


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


eStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)


history = model.fit(training_set,
                              epochs=60,
                              validation_data=validation_set,
                              verbose=1,
                              callbacks=[eStop])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

# Grafik başlığını ve etiketlerini belirleyin
plt.title('Model Doğruluk')
plt.ylabel('Doğruluk')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Grafik dosyasını kaydedin
plt.savefig('accuracy_graph.png')
plt.clf()  # Grafik penceresini temizle

# Kayıp (loss) metriklerini çizdirin
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# Grafik başlığını ve etiketlerini belirleyin
plt.title('Model Kayıp')
plt.ylabel('Kayıp')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Grafik dosyasını kaydedin
plt.savefig('loss_graph.png')


score = model.evaluate(validation_set)


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(img_width, img_height),
                                            color_mode='rgb',
                                            batch_size=128,
                                            class_mode='categorical',
                                            shuffle=False)
score = model.evaluate(test_set)
model.save("plane_detector-new")