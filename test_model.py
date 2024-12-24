from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model("breast_cancer_cnn_model.h5")

# Path to a test image
test_image_path = r"C:\Users\hp\OneDrive\Bureau\dataset\malignant\mdb130.pgm"

# Load and preprocess the test image
image = Image.open(test_image_path).convert("RGB")
image = image.resize((224, 224))
image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension

# Predict
predictions = model.predict(image_array)
class_names = ['benign', 'malignant', 'normal']
predicted_class = class_names[np.argmax(predictions)]
print(f"Predicted Class: {predicted_class}")
