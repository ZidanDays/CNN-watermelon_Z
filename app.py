import os
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))




def load_custom_model():
    # model_path = "/model.h5"  # Ubah sesuai dengan nama file dan path model Anda
    # model = load_model(model_path)
    model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
    dataset_path = "plant_dataset/"
    path_gambar_sehat = os.path.join(dataset_path, "sehat")
    path_gambar_tidak_sehat = os.path.join(dataset_path, "tidak_sehat")
    return model



def detect_health(image):
    # Logika deteksi kesehatan tanaman semangka di sini
    # Misalnya, menggunakan model yang telah dilatih untuk klasifikasi gambar

    # Contoh sederhana
    if image.size[0] * image.size[1] > 1000:
        return "Tanaman sehat"
    else:
        return "Tanaman tidak sehat"

def main():
    st.title("Deteksi Kesehatan Tanaman Semangka")

    st.write("Unggah gambar tanaman semangka untuk mendeteksi kesehatannya.")

    # Upload gambar
    uploaded_image = st.file_uploader("Unggah gambar tanaman semangka", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar tanaman semangka", use_column_width=True)

        # Deteksi kesehatan tanaman semangka
        health_status = detect_health(image)
        st.write("Status kesehatan tanaman:", health_status)

        # Memuat model
        model = load_custom_model()

        # Menyiapkan dataset
        train_dir = "path/to/train_dataset"
        validation_dir = "path/to/validation_dataset"

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
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )

        # Melatih model
        history = model.fit(
            train_generator,
            steps_per_epoch=100,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=50
        )

if __name__ == "__main__":
    main()
