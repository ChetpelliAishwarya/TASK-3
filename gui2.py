import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\chais\Desktop\CAR DETECTION\my_model.h5', custom_objects={
    'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy,
    'MeanSquaredError': tf.keras.losses.MeanSquaredError,
    'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError
})

# Load annotations from JSON file
with open(r'C:\Users\chais\Desktop\CAR DETECTION\via_region_data (1).json', 'r') as f:
    annotations = json.load(f)

def predict(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img)
    color_pred, count_pred, people_pred = predictions
    color_idx = np.argmax(color_pred, axis=1)[0]
    count = round(count_pred[0][0])
    males = round(people_pred[0][0])
    females = round(people_pred[0][1])
    
    # Map index to color
    color_map = {0: 'red', 1: 'blue'}
    color = color_map.get(color_idx, 'other')

    return color, count, males, females

def get_annotations(filename, annotations):
    # Define a color mapping for red and blue swapping
    color_map = {'red': 'blue', 'blue': 'red'}
    
    # Find the annotation entry for the given filename
    for key, value in annotations.items():
        if value['filename'] == filename:
            region = value['regions'][0]['region_attributes']
            
            # Extract car colors and split them
            car_colors = region['car color'].split('\n')
            
            # Filter and map car colors
            annotated_colors = [color_map.get(color.strip().lower(), None) for color in car_colors if color.strip().lower() in color_map]
            
            # Extract car count and people count
            car_count = int(region['car count'])
            people_count = region['people count'].split('\n')
            males = int(people_count[0].split(':')[1])
            females = int(people_count[1].split(':')[1])
            
            return annotated_colors, car_count, males, females
            
    return None

# Streamlit UI
st.title('Traffic Signal Image Analysis')
st.write("Upload an image to predict the car color, car count, and number of people.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    image = cv2.imread("temp.jpg")
    st.image(image, channels="BGR")
    
    

    # Get annotations
    annotations_data = get_annotations(uploaded_file.name, annotations)
    if annotations_data:
        car_colors, car_count, males, females = annotations_data
        st.write(f" Car Colors: {', '.join(car_colors)}")
        st.write(f"Car Count: {car_count}")
        st.write(f" Males: {males}")
        st.write(f"Females: {females}")
    else:
        st.write("No annotations found for this image.")
