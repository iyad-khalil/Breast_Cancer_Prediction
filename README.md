# Breast Cancer Prediction

## Project Overview
The **Breast Cancer Prediction** project is a deep learning-based web application that detects and classifies breast cancer from mammogram images. The project leverages a Convolutional Neural Network (CNN) model trained on mammogram images and deployed using Flask to provide an intuitive web interface for users to upload medical images and receive predictions.

## Features
- **Model Training:** Custom CNN model for classifying images into benign, malignant, or normal categories.
- **Web Interface:** User-friendly interface for image upload and prediction display.
- **Automated Preprocessing:** Image normalization and resizing for better model performance.
- **Prediction Confidence:** Provides classification results with confidence percentages.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iyad-khalil/Breast_Cancer_Prediction.git
   cd Breast_Cancer_Prediction
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Model Training

To train the model 

1. Run the training script:
   ```bash
   python train_model.py
   ```
2. The trained model will be saved as `breast_cancer_cnn_model.h5`.

## Usage

1. **Run the web application:**
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   Open your browser and go to `http://127.0.0.1:5000`.

3. **Upload an image:**
   - Click on the upload section and select a mammogram image.
   - Click the "Analyze Image" button to get the prediction.

## File Descriptions

- **`app.py`**: Main Flask application that loads the trained model and handles image uploads.
- **`preprocess.py`**: Prepares the dataset by loading and processing images.
- **`train_model.py`**: Defines and trains the CNN model using TensorFlow.
- **`test_model.py`**: Tests the trained model on a sample image.
- **`index.html`**: Frontend UI to interact with the system.

## Dependencies

Ensure the following dependencies are installed:

```txt
Flask
TensorFlow
Pillow
numpy
matplotlib
```

## Example Prediction

1. Upload a sample mammogram image.
2. View the result indicating whether the image is classified as benign, malignant, or normal.
3. The confidence level of the prediction will also be displayed.

## Future Enhancements

- Improve model accuracy with more data augmentation.
- Add more diagnostic metrics.
- Deploy the application to cloud services.

## Acknowledgements

This project is inspired by research in medical imaging and AI for early cancer detection.

---

**Author:** KHALIL Iyad
 

