import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Fungsi untuk memuat model deteksi kesehatan tanaman semangka
def load_model():
    # model = load_model("plant_health_detection_model.h5")  # Ganti dengan path tempat model Anda disimpan
    model = load_model("model")  # Ganti dengan path tempat model Anda disimpan
    return model

# Fungsi untuk mendeteksi kesehatan tanaman semangka menggunakan model
def detect_health_with_model(image, model):
    # Lakukan pra-pemrosesan gambar jika diperlukan
    # Misalnya, ubah ukuran gambar, normalisasi, dll.
    # ...
    
    # Lakukan prediksi menggunakan model
    prediction = model.predict(image)
    
    # Konversi hasil prediksi menjadi label kategori
    if prediction > 0.5:
        return "Tanaman sehat"
    else:
        return "Tanaman tidak sehat"

def main():
    st.title("Deteksi Kesehatan Tanaman Semangka")
    
    # Muat model
    model = load_model()
    
    st.write("Unggah gambar tanaman semangka untuk mendeteksi kesehatannya.")

    # Upload gambar
    uploaded_image = st.file_uploader("Unggah gambar tanaman semangka", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_image)
        st.image(image, caption="Gambar tanaman semangka", use_column_width=True)

        # Ubah gambar menjadi format yang dapat diterima oleh model
        # Misalnya, ubah ke bentuk array numpy dan lakukan pra-pemrosesan
        # ...

        # Deteksi kesehatan tanaman semangka menggunakan model
        health_status = detect_health_with_model(image, model)
        st.write("Status kesehatan tanaman:", health_status)

if __name__ == "__main__":
    main()
