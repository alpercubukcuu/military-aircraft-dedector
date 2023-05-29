from keras.models import load_model
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("plane_detector-new")

# Load the image
img = Image.open('splitted_dataset_v3/test/F-35_JSF/aug_0_298.png').convert('RGB').resize((224, 224))

# Convert the image to a numpy array and preprocess it
data = np.array(img)
data = data.reshape(-1, 224, 224, 3)
data = preprocess_input(data)

# Predict the class of the image
prediction = model.predict(data)

# Use the training class indices to interpret the prediction
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
training_set = train_datagen.flow_from_directory('splitted_dataset_v3/train', target_size=(224, 224), color_mode='rgb', batch_size=128, class_mode='categorical')
class_names = list(training_set.class_indices.keys()) # get the class labels
class_names.sort() # make sure it's in the correct order

result = np.argmax(prediction[0])
print(class_names[result])