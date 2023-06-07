from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os
import openai

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join("uploads", filename))
            return redirect(url_for('predict', filename=filename))
    return render_template('upload.html')

openai.api_key = '****'

def get_plane_info(plane_name):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"{plane_name} uçağı hakkında türkçe bilgi verir misin ? ",
      temperature=0.5,
      max_tokens=100
    )
    return response.choices[0].text.strip()



@app.route('/predict/<filename>')
def predict(filename):
    # Load the trained model
    model = load_model("plane_detector-new")

    # Load the image
    img = Image.open('uploads/' + filename).convert('RGB').resize((224, 224))

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

    # Get detailed information about the predicted plane from GPT-3
    # plane_info = get_plane_info(class_names[result])

    return class_names[result]

if __name__ == "__main__":
    app.run(debug=True)
