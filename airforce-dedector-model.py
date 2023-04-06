from keras.models import load_model
from keras.applications.resnet import preprocess_input
from PIL import Image
import numpy as np


model = load_model("planededector")

img = Image.open('airplane-dataset-trans/test/DC-4/5-614.jpg').resize((224, 224))
data = np.array(img)

print(data.shape, data.ndim)

data = data.reshape(-1, 224, 224, 3)

print(data.shape, data.ndim)

data = preprocess_input(data)

print(data, data.shape)

prediction = model.predict(data)


img_classes = ["A-10_Thunderbolt", "Airliner", "ATR_72_ASW", "ATR-72_Airliner",
               "B-1_Lancer", "B-2_Spirit", "B-29_Superfortress", "B-52_Stratofortress",
               "B-57_Canberra", "BusinessJet", "C-5_Galaxy", "C-17_Globemaster",
               "C-40_Clipper", "C-130_Hercules", "C-135_Stratolifter", "C-295M_CASA_EADS",
               "DC-4", "DC-4E", "E-2_Hawkeye", "E-3_Sentry",
               "EA-6B_Prowler", "F-4_Phantom", "F-15_Eagle", "F-16_Falcon",
               "F-18_Hornit", "F-22_Raptor", "F-35_JSF", "KC-767_Tanker",
               "King_Air_Beechcraft_Airliner", "King_Air_Beechcraft_ISR", "LightACHighSetWing", "LightACLowSetWing",
               "LightACTwinEnginProp", "P-3_Orion", "RC-135_Rivit_Joint", "Su-37_Flanker",
               "T-1A_Jayhawk_Trainer", "T-43A_Boeing737-253A_Trainer", "Tu-160_Tupolev_White_Swan", "UTA_Fokker_50_Utility_Transport"]


result = np.argmax(prediction[0])
print(img_classes[result])