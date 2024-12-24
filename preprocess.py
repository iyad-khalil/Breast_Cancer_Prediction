import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to your dataset folder
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
print("Loading dataset...")
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

# Visualize some images
print("Visualizing data...")
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i]])
    plt.axis("off")
plt.show()

print("Preprocessing complete!")
