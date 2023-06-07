import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array, save_img

train_files = "airplane-dataset-trans/test/"
augmented_files = "airplane-dataset-trans/augmented_test/"
os.makedirs(augmented_files, exist_ok=True)

# Veri artırma yöntemleri
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

for subdir, dirs, files in os.walk(train_files):
    for file in files:
        img_path = os.path.join(subdir, file)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        i = 0
        subdir_name = subdir.split('/')[-1]
        os.makedirs(os.path.join(augmented_files, subdir_name), exist_ok=True)

        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=os.path.join(augmented_files, subdir_name),
                                        save_prefix='aug', save_format='png'):
            i += 1
            if i >= 3:  # Sadece 3 kat artırılmış görüntü oluşturuyoruz
                break