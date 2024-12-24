import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers, models

# --- Preprocessing Section ---
# Path to your dataset
dataset_path = r"C:\Users\hp\OneDrive\Bureau\dataset"

# Image settings
batch_size = 32
img_size = (224, 224)

# Helper function to load .pgm images and labels
def load_pgm_dataset(dataset_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))  # Folders: benign, malignant, normal
    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        for file_name in os.listdir(class_path):
            if file_name.endswith('.pgm'):
                file_path = os.path.join(class_path, file_name)
                # Load the image, resize, and normalize pixel values
                image = Image.open(file_path).convert("RGB")  # Convert PGM to RGB
                image = image.resize(img_size)
                image = np.array(image) / 255.0  # Normalize to [0, 1]
                images.append(image)
                labels.append(label)
    return np.array(images), np.array(labels), class_names

# Load images and labels
images, labels, class_names = load_pgm_dataset(dataset_path)

# Shuffle and split into training and validation
dataset_size = len(images)
indices = np.arange(dataset_size)
np.random.shuffle(indices)
split = int(0.8 * dataset_size)

train_images, val_images = images[indices[:split]], images[indices[split:]]
train_labels, val_labels = labels[indices[:split]], labels[indices[split:]]

# Convert to TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)

# --- Model Definition Section ---
# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes: benign, malignant, normal
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Training Section ---
# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # Adjust epochs based on time available
)

# --- Evaluation Section ---
# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# --- Saving Section ---
# Save the trained model
model.save("breast_cancer_cnn_model.h5")
print("Model saved as breast_cancer_cnn_model.h5")
