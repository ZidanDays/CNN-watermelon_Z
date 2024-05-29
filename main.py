import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('CNN_model.hdf5')
    return model

model = load_model()

st.write("""
         # Cat or Dog Classification
         """
         )


file = st.file_uploader("Please upload an image of a cat or dog", type=["jpg", "png"])

import numpy as np
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
    size = (64, 64)
    user_image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    user_image = np.asarray(user_image)
    user_image = np.expand_dims(user_image, axis = 0)
    result = model.predict(user_image)
    
    return result

print(import_and_predict)

if file is None:
    st.error("Please upload an image file")
else:
    
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Dog', 'Cat']
    if prediction[0][0] == 1:
        st.success('this image most is likely a Dog')
    else:
        st.success('this image is most likely a Cat')