import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Function to load the model
def load_custom_model():
    # model_path = "CNN_model.hdf5"  # Path to your trained model
    model_path = "CNN_model.h5"  # Path to your trained model
    model = load_model(model_path)
    return model

# Function to make prediction
def predict(image, model):
    # Preprocess the image
    image = np.array(image.resize((64, 64))) / 255.0  # Resize and normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image)
    
    # Get class label
    class_label = np.argmax(prediction)
    
    # Map class index to class label
    class_labels = ["Anthracnose", "cercospora", "Downy_Mildew", "Healthy", "layu_fusarium", "Mosaic_Virus"]
    plant_health_status = class_labels[class_label]
    
    return plant_health_status


# Main function to run the Streamlit app
def main():
    st.title("Watermelon Plant Health Detection")
    st.write("Upload an image of a watermelon plant to detect its health.")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded image", use_column_width=True)

        # Load model
        model = load_custom_model()

        # Make prediction
        health_status = predict(image, model)

        # Display result
        st.write("Plant health status:", health_status)

# Run the main function
if __name__ == "__main__":
    main()


