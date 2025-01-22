import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Initialize Flask app
app = Flask(__name__)

# Constants
IMG_SIZE = (256, 256)
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load the model
def load_model_with_custom_objects():
    try:
        # Define custom metrics if needed
        custom_objects = {
            'mse': tf.keras.metrics.MeanSquaredError(),
            'accuracy': tf.keras.metrics.Accuracy()
        }
        
        MODEL_PATH = "path_to _your_modal
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects,
            compile=False  # Load without compilation to avoid metric issues
        )
        # Recompile the model with appropriate metrics
        model.compile(
            optimizer='adam',
            loss=['mse', 'sparse_categorical_crossentropy'],
            metrics={
                'age_output': ['mse'],
                'gender_output': ['accuracy']
            }
        )
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

# Load the model
model = load_model_with_custom_objects()

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not loaded properly.")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Resize to target size
        resized = cv2.resize(enhanced_rgb, IMG_SIZE)
        
        # Normalize to [0, 1]
        return resized / 255.0
    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
        return "Error: Model could not be loaded. Please check the model file and path."
        
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file uploaded!"
            
        file = request.files['file']
        if file.filename == '':
            return "No file selected!"
            
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        if processed_image is None:
            return "Error processing the image. Please upload a valid image."
            
        # Add batch dimension
        processed_image = np.expand_dims(processed_image, axis=0)
        
        try:
            # Predict
            predictions = model.predict(processed_image)
            age_pred = predictions[0][0]
            gender_pred = np.argmax(predictions[1], axis=1)[0]
            
            # Map gender prediction to label
            gender_label = {0: "Male", 1: "Female", 2: "Unknown"}
            predicted_gender = gender_label.get(gender_pred, "Unknown")
            
            return f"Predicted Age: {age_pred:.2f}, Predicted Gender: {predicted_gender}"
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return f"Error making prediction: {str(e)}"
    
    return '''
        <!doctype html>
        <title>Age and Gender Prediction</title>
        <h1>Upload an Image</h1>
        <form method="post" enctype="multipart/form-data">
          <input type="file" name="file">
          <input type="submit" value="Upload">
        </form>
    '''

if __name__ == "__main__":
    app.run(debug=True)
