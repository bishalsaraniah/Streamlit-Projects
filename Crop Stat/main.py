import os
import json
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
import tensorflow as tf
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Streamlit App
st.set_page_config(page_title="Crop Disease Detector", layout="wide")
st.title('Crop Disease Detection')

# Get API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

# Set working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# List of crop models
crop_models = ["apple", "corn", "cherry", "peach", "pepper", "potato", "strawberry", "grape", "tomato"]

# User selects a crop type
crop_selection = st.selectbox("Select the crop type:", crop_models)

# Load only the selected model
@st.cache_resource
def load_model(crop_name):
    model_path = os.path.join(working_dir, f"model/{crop_name}.h5")
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

# Load class indices for all crops
@st.cache_data
def load_class_indices():
    indices_path = os.path.join(working_dir, "class_indices.json")
    if os.path.exists(indices_path):
        return json.load(open(indices_path))
    else:
        return None

# Load model and class indices
selected_model = load_model(crop_selection)
all_class_indices = load_class_indices()

# Function to preprocess image
def load_and_preprocess_image(image, target_size=(224, 224)):
    if isinstance(image, np.ndarray):
        # If input is a numpy array (from camera)
        img = Image.fromarray(image)
    else:
        # If input is a file upload
        img = Image.open(image)
    
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize
    return img_array, img

# Function to predict the class
def predict_image_class(model, image, crop_name, all_indices):
    if all_indices is None or crop_name not in all_indices:
        return "Error: Class indices not found for this crop", None
    
    # Get the class indices specific to the selected crop
    crop_indices = all_indices[crop_name]
    preprocessed_img, original_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Convert to string since JSON keys are strings
    predicted_class_key = str(predicted_class_index)
    if predicted_class_key in crop_indices:
        return crop_indices[predicted_class_key], original_img
    else:
        return f"Unknown class index: {predicted_class_index}", None

# Function to get treatment recommendations from Gemini
def get_treatment_recommendation(crop_type, disease_name, image=None):
    try:
        if not GEMINI_API_KEY:
            return "GEMINI_API_KEY not found in environment variables. Please check your .env file."
        
        # Configure the model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare the prompt
        prompt = f"Provide detailed treatment recommendations for {disease_name} in {crop_type} plants. Include:" + \
                 "\n1. Brief description of the disease" + \
                 "\n2. Chemical treatments (conventional)" + \
                 "\n3. Organic treatments" + \
                 "\n4. Preventive measures" + \
                 "\n5. Environmental conditions to avoid"
        
        # If we have an image, we'll use multimodal capabilities
        if image:
            # Convert PIL image to bytes for Gemini
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            st.subheader("Treatment Recommendations")
            response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": img_byte_arr}])
        else:
            response = model.generate_content(prompt)
            
        return response.text
    except Exception as e:
        return f"Error getting treatment recommendation: {str(e)}"

# Create tabs for different image input methods
tab1, tab2 = st.tabs(["Upload Image", "Capture Image"])
image = None
prediction_placeholder = st.empty()
treatment_placeholder = st.empty()

# Tab 1: Upload Image
with tab1:
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display the image
        col1, col2 = st.columns([1, 2])
        with col1:
            display_img = Image.open(uploaded_image)
            resized_img = display_img.resize((250, 250))
            st.image(resized_img, caption="Uploaded Image")
            
            if st.button('Analyze Uploaded Image'):
                image = uploaded_image

# Tab 2: Capture Image
with tab2:
    st.markdown("### Capture Image from Camera")
    
    # Camera input
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Display the image
        col1, col2 = st.columns([1, 2])
        with col1:
            display_img = Image.open(camera_image)
            resized_img = display_img.resize((250, 250))
            st.image(resized_img, caption="Captured Image")
            
            if st.button('Analyze Captured Image'):
                image = camera_image

# Process the image if we have one
if image is not None:
    with st.spinner('Classifying disease...'):
        if selected_model and all_class_indices:
            st.info(f'Using model for: {crop_selection}')
            prediction, img_for_gemini = predict_image_class(selected_model, image, crop_selection, all_class_indices)
            
            # Display prediction
            if "Error" not in prediction and "Unknown" not in prediction:
                st.success(f'Detected Disease: {prediction}')
                
                # Get treatment recommendations from Gemini
                with st.spinner('Getting treatment recommendations...'):
                    treatment = get_treatment_recommendation(crop_selection, prediction, img_for_gemini)
                    
                    # Display the treatment recommendations
                    st.markdown(treatment)
            else:
                st.error(prediction)
        else:
            if not selected_model:
                st.error(f"Model not found for {crop_selection}.")
            if not all_class_indices:
                st.error(f"Class indices not found.")
            elif crop_selection not in all_class_indices:
                st.error(f"Class indices for {crop_selection} not found.")

# Add instructions at the bottom
st.markdown("---")
st.markdown("""
### How to use this app:
1. Select the crop type from the dropdown menu
2. Either upload an image of the crop or take a picture using your camera
3. Click on 'Analyze' to detect the disease
4. View the disease prediction and treatment recommendations
5. Optionally save the results for future reference
""")

# Add app info in the sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app detects diseases in crops using machine learning models and provides treatment recommendations using Gemini AI.
    
    Supported crops:
    - Apple
    - Corn
    - Cherry
    - Peach
    - Pepper
    - Potato
    - Strawberry
    - Grape
    - Tomato
    """)