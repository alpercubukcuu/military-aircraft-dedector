from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet import ResNet50
import matplotlib.pyplot as plt


train_files = "airplane-dataset-trans/train/"
test_files = "airplane-dataset-trans/test/"

img = load_img(test_files + "A-10_Thunderbolt/image (198).png")
print(img_to_array(img).shape)
plt.imshow(img)
# plt.show() #Deneme Görseli


# Veri arttırma yöntemleri
train_data = ImageDataGenerator().flow_from_directory(train_files, target_size=(224, 224))
test_data = ImageDataGenerator().flow_from_directory(test_files, target_size=(224, 224))
numberOfAirplaneType = 40

# ResNet50 modelinin çıkış şeklini alır
output_shape = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).output_shape[1:]

# Model oluşturma
model = Sequential()
model.add(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
model.add(GlobalAveragePooling2D())

# Öğrenme işlemini sadece son eklenen katmanda gereçekleşmesi lazım
for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=numberOfAirplaneType, activation='softmax'))
print(model.summary())

# Modeli derleme
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

# Modeli eğitme
batch_size = 4
model.fit(train_data,
          steps_per_epoch=400//batch_size,
          epochs=20,
          validation_data=test_data,
          validation_steps=200//batch_size)


model.save("planededector")