# app.py
import keras
import streamlit as st
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
import numpy as np
from PIL import Image

# Load the trained model with custom objects if needed
custom_objects = {
    'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy
}

# Load the trained model
model = tf.keras.models.load_model('my_model.h5',custom_objects=custom_objects)

# Define class labels 
class_labels = {0: 'Potato___Early_blight', 1: 'Potato___Late_blight', 2:'Potato___healthy' }   

# Define the Streamlit app
st.title("Potato Plant Disease Classification")

# Upload an image
uploaded_file = st.file_uploader("Please upload potato plant leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img_array = np.array(image.resize((256, 256)))  # Resize image to match model input size
    img_array = img_array / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = round(100 * (np.max(predictions[0])), 2)

    # Display results
    st.write(f"The provided plant is most likely : {predicted_class}")
    st.write(f"The confidence level is : {confidence}")


