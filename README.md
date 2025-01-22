# DermaNet: Age and Gender Prediction from Skin Images


Welcome to **DermaNet**! A deep learning model designed to predict the age and gender of individuals based on skin lesion images. Powered by advanced Convolutional Neural Networks (CNNs), DermaNet provides accurate and reliable predictions that can be useful for dermatology research, personalized skincare, and more.

## 🚀 Key Features

- **Age Prediction**: DermaNet estimates the approximate age of individuals based on their skin images, allowing for age-related analysis.
- **Gender Prediction**: It predicts the gender of individuals (male, female, or unknown), providing valuable demographic insights.
- **High-Quality Model**: Built on a robust CNN architecture with preprocessing techniques like CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian Blurring for enhanced image quality.
- **User-Friendly Interface**: Easy-to-use scripts for prediction and analysis, making it accessible to researchers and practitioners alike.

## 🧠 Model Architecture
DermaNet utilizes a state-of-the-art Convolutional Neural Network (CNN) for feature extraction and prediction tasks:

- **Convolutional Layers**: Learn intricate patterns in skin images.
- **Max-Pooling Layers**: Reduce dimensionality and computational cost.
- **Batch Normalization**: Speed up training and improve generalization.
- **Fully Connected Layers**: Perform the final predictions for age and gender.
- **Dropout Regularization**: Reduce overfitting and improve robustness.

## 🖼️ How It Works

1. **Image Preprocessing**: Images are enhanced using CLAHE for better contrast, resized to 256x256 pixels, and normalized for deep learning.
2. **Model Training**: The model is trained on a large dataset of annotated skin images, learning the correlation between age, gender, and skin features.
3. **Prediction**: After training, the model predicts the age and gender of new images based on learned patterns.

## 📊 Results & Performance

- **Age Prediction**: Achieved a low Mean Absolute Error (MAE) for precise age estimation.
- **Gender Prediction**: High classification accuracy for predicting gender (male, female, or unknown).

## 📂 How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/Alokbpandey/DermaNet.git
cd DermaNet
```

### 2. Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
```

### 3. Load the Pre-trained Model
The pre-trained model is saved as `DermaNet.h5`. You can load it as follows:
```python
from tensorflow.keras.models import load_model
model = load_model('DermaNet.h5')
```

### 4. Make Predictions
Predict the age and gender of a new image:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

age_pred, gender_pred = model.predict(img_array)
print(f"Predicted Age: {age_pred}")
print(f"Predicted Gender: {['Male', 'Female', 'Unknown'][np.argmax(gender_pred)]}")
```

## ⚙️ Technologies Used

- **TensorFlow/Keras**: For building and training the deep learning model.
- **OpenCV**: For image preprocessing and enhancement.
- **Matplotlib**: For visualizing training performance.
- **NumPy**: For efficient numerical computations.

## 📈 Training History

- **Age MAE**: The training and validation Mean Absolute Error (MAE) for age prediction.
- **Gender Accuracy**: The training and validation accuracy for gender classification.

## 🚧 Limitations

- **Image Quality**: Performance may degrade with poor-quality or distorted images.
- **Age Range**: Predictions are approximate and may not always be precise for every individual.
- **Dataset Bias**: Results depend on the diversity and quality of the training dataset.

## 🧑‍🤝‍🧑 Contribute

We welcome contributions! Feel free to fork the repository, open issues, or submit pull requests to help enhance DermaNet further.

### Suggestions for Contributions:
- Expand the training dataset with diverse images.
- Improve preprocessing techniques for low-quality images.
- Optimize the model architecture for faster inference.
- Integrate additional demographic predictions.

## 💬 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Start predicting with **DermaNet** today and unlock insights into age and gender from skin images!
