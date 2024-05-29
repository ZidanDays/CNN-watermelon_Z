import os
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))


def load_custom_model():
    # Load the InceptionV3 model without the top classification layer
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # Add custom classification layers on top of the base model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)  # Assuming binary classification (healthy vs unhealthy)
    
    # Combine the base model with custom layers
    model = Model(inputs=base_model.input, outputs=output)
    
    return model


def detect_health(image):
    # Custom logic for detecting watermelon plant health
    # For example, using the trained model for image classification
    
    # Placeholder logic
    if image.size[0] * image.size[1] > 1000:
        return "Healthy"
    else:
        return "Unhealthy"


def main():
    st.title("Watermelon Plant Health Detection")

    st.write("Upload an image of a watermelon plant to detect its health.")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image of a watermelon plant", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Watermelon plant image", use_column_width=True)

        # Detect watermelon plant health
        health_status = detect_health(image)
        st.write("Plant health status:", health_status)

        # Load model
        model = load_custom_model()

        # Prepare dataset directories
        train_dir = "path/to/train_dataset"
        validation_dir = "path/to/validation_dataset"

        # Prepare dataset (if using a pre-existing model, this can be skipped)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary'
        )

        # Compile model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model
        history = model.fit(
            train_generator,
            steps_per_epoch=100,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=50
        )

if __name__ == "__main__":
    main()
