from asyncio import run
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model


import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
import numpy as np

# Load the trained model
try:
    model = tf.keras.models.load_model('pneumonia_detection_model_2.h5')  # Updated model file
except (OSError, IOError) as e:
    st.error("Error loading model. Please ensure the model file is in the correct location.")
    st.stop()

# Define a function to predict pneumonia
def predict_pneumonia(img, model):
    try:
        img = img.convert("RGB")  # Ensure the image has 3 channels (RGB)
        img = img.resize((150, 150))  # Resize the image to match the input shape of the model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image
        prediction = model.predict(img_array)
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# Streamlit web app layout
st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to detect pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Chest X-ray Image', use_column_width=True)

        if st.button("Predict"):
            prediction = predict_pneumonia(img, model)
            
            if prediction is not None:
                prediction_percentage = prediction * 100  
                
                if prediction > 0.5:
                    st.success(f"### Result: Pneumonia Positive with {prediction_percentage:.2f}% confidence")
                else:
                    st.success(f"### Result: Pneumonia Negative with {100 - prediction_percentage:.2f}% confidence")
            else:
                st.error("Prediction failed. Please try again.")
    except UnidentifiedImageError:
        st.error("Invalid image format. Please upload a valid chest X-ray image.")
