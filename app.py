from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "breast_cancer_cnn_model.h5"
model = load_model(MODEL_PATH)

# Class names
class_names = ['benign', 'malignant', 'normal']

# Preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        if file:
            # Preprocess the image
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)

            # Predict using the model
            predictions = model.predict(processed_image)
            confidence = np.max(predictions) * 100  # Confidence percentage
            predicted_class = class_names[np.argmax(predictions)]

            # Return prediction
            return render_template('index.html', prediction=predicted_class, confidence=confidence)

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
