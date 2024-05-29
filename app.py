import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
from efficientnet.tfkeras import EfficientNetB0

# Load the model
model_path = './models/v2_new_arch.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define the class labels
label = ['Aerotitis Barotrauma', 'Cerumen', 'Corpus Alienum', 'M Timpani normal', 'Myringitis Bulosa', 'Normal', 'OE Difusa', 'OE Furunkulosa', 'OMA Hiperemis', 'OMA Oklusi Tuba', 'OMA Perforasi', 'OMA Resolusi', 'OMA Supurasi', 'OMed Efusi', 'OMedK Resolusi', 'OMedK Tipe Aman', 'OMedK Tipe Bahaya', 'Otomikosis', 'Perforasi Membran Tympani', 'Tympanosklerotik']

# Data for file counts
file_counts = {
    'OMed Efusi': 26,
    'Corpus Alienum': 6,
    'OMedK Tipe Aman': 132,
    'OMA Resolusi': 12,
    'M Timpani normal': 161,
    'OMA Hiperemis': 79,
    'Tympanosklerotik': 55,
    'OMA Oklusi Tuba': 58,
    'OE Difusa': 95,
    'Myringitis Bulosa': 10,
    'OMA Supurasi': 55,
    'Perforasi Membran Tympani': 10,
    'OMedK Resolusi': 23,
    'Cerumen': 55,
    'OMA Perforasi': 45,
    'OMedK Tipe Bahaya': 94,
    'Otomikosis': 31,
    'OE Furunkulosa': 14,
    'Aerotitis Barotrauma': 10,
    'Normal': 132,
}


# Streamlit app
st.title('Ear Disease Classification Demo')

# Sidebar menu for classification or datasets version
menu_option = st.sidebar.radio("Menu", ("Classification", "Datasets Version"))

if menu_option == "Classification":
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Make predictions
        predictions = model.predict(processed_image)

        # Get the predicted class index and probability
        predicted_class_index = np.argmax(predictions)
        predicted_probability = predictions[0][predicted_class_index]

        # Display the predicted class and probability
        st.write(f'Predicted Class: {label[predicted_class_index]}')
        st.write(f'Predicted Probability: {predicted_probability:.4f}')

elif menu_option == "Datasets Version":
    # Convert file counts to DataFrame for table display
    file_counts_df = pd.DataFrame(list(file_counts.items()), columns=['Class', 'Image Count'])
    st.write("Dataset V.01")
    st.write("File Counts for Each Class (raw data):")
    st.table(file_counts_df)
