from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
from tensorflow import keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model_path = "C:/Users/rithu/Downloads/ML/madhuflask/plant_disease_prediction_model.h5"
loaded_model = keras.models.load_model(model_path)

# Class names dictionary
class_indices = {
    "0": "Apple:Apple_scab",
    "1": "Apple:Black_rot",
    "2": "Apple:Cedar_apple_rust",
    "3": "Apple:healthy",
    "4": "Blueberry:healthy",
    "5": "Cherry_(including_sour):Powdery_mildew",
    "6": "Cherry_(including_sour):healthy",
    "7": "Corn_(maize):Cercospora_leaf_spot Gray_leaf_spot",
    "8": "Corn_(maize):Common_rust",
    "9": "Corn_(maize):Northern_Leaf_Blight",
    "10": "Corn_(maize):healthy",
    "11": "Grape:Black_rot",
    "12": "Grape:Esca_(Black_Measles)",
    "13": "Grape:Leaf_blight_(Isariopsis_Leaf_Spot)",
    "14": "Grape:healthy",
    "15": "Orange:Haunglongbing_(Citrus_greening)",
    "16": "Peach:Bacterial_spot",
    "17": "Peach:healthy",
    "18": "Pepper,_bell:Bacterial_spot",
    "19": "Pepper,_bell:healthy",
    "20": "Potato:Early_blight",
    "21": "Potato:Late_blight",
    "22": "Potato:healthy",
    "23": "Raspberry:healthy",
    "24": "Soybean:healthy",
    "25": "Squash: Powdery_mildew",
    "26": "Strawberry:Leaf_scorch",
    "27": "Strawberry:healthy",
    "28": "Tomato:Bacterial_spot",
    "29": "Tomato:Early_blight",
    "30": "Tomato:Late_blight",
    "31": "Tomato:Leaf_Mold",
    "32": "Tomato:Septoria_leaf_spot",
    "33": "Tomato:Spider_mites Two-spotted_spider_mite",
    "34": "Tomato:Target_Spot",
    "35": "Tomato:Tomato_Yellow_Leaf_Curl_Virus",
    "36": "Tomato:Tomato_mosaic_virus",
    "37": "Tomato: healthy"
}

# Preprocess function to resize and normalize the image
def preprocess_image(img):
    # Resize the image to match the input size expected by the model
    img = img.resize((224, 224))  # Adjust dimensions according to your model
    # Convert image to numpy array and normalize pixel values
    img_array = np.array(img) / 255.0
    return img_array

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Convert the image file to a PIL Image object
    img = Image.open(io.BytesIO(file.read()))

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make the prediction
    prediction = loaded_model.predict(np.expand_dims(img_array, axis=0))

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Get the predicted class name using class_indices dictionary
    predicted_class_name = class_indices[str(predicted_class_index)]

    result = "The predicted class is: {}".format(predicted_class_name)

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
