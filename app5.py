# # # # # import streamlit as st
# # # # import numpy as np
# # # # from PIL import Image
# # # # import tensorflow as tf
# # # # import streamlit as st

# # # # # Load trained model
# # # # model = tf.keras.models.load_model('leaf_disease_classifier.h5')

# # # # # Streamlit App
# # # # st.title("Sistem Pendeteksi Penyakit Tanaman Semangka")

# # # # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# # # # if uploaded_file is not None:
# # # #     try:
# # # #         # Load the image
# # # #         image = Image.open(uploaded_file)
        
# # # #         # Convert image to RGB if it's not already
# # # #         if image.mode != "RGB":
# # # #             image = image.convert("RGB")
        
# # # #         # Display the uploaded image
# # # #         st.image(image, caption='Uploaded Image.', use_column_width=True)
# # # #         st.write("")
# # # #         st.write("Classifying...")

# # # #         # Preprocess the image for prediction
# # # #         img = image.resize((150, 150))  # Resize the image to the desired dimensions
# # # #         img_array = np.array(img)  # Convert the image to an array
# # # #         img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch
# # # #         img_array = img_array / 255.0  # Normalize the image data

# # # #         # Make prediction
# # # #         prediction = model.predict(img_array)
# # # #         predicted_class_index = np.argmax(prediction)
# # # #         confidence = prediction[0][predicted_class_index]
# # # #         confidence_threshold = 0.3  # Lowering the threshold for confidence

# # # #         class_list = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']

# # # #         if confidence >= confidence_threshold:
# # # #             predicted_class = class_list[predicted_class_index]
# # # #             # Display the predicted class
# # # #             st.write(f"Predicted class: {predicted_class}")
# # # #         else:
# # # #             # Display confidences for all classes in the error message
# # # #             confidence_str = ", ".join([f"{class_list[i]}: {pred}" for i, pred in enumerate(prediction[0])])
# # # #             st.write(f"Gambar tidak valid untuk pendeteksian penyakit tanaman semangka. Silakan coba dengan gambar lain. Confidences: {confidence_str}")

# # # #     except Exception as e:
# # # #         st.write(f"Error: {e}")

# # # # else:
# # # #     st.write("Silakan unggah gambar.")

# # # import streamlit as st
# # # import numpy as np
# # # from PIL import Image
# # # import tensorflow as tf

# # # # Load trained model
# # # model = tf.keras.models.load_model('leaf_disease_classifier.h5')

# # # # Streamlit App
# # # st.title("Sistem Pendeteksi Penyakit Tanaman Semangka")

# # # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# # # if uploaded_file is not None:
# # #     try:
# # #         # Load the image
# # #         image = Image.open(uploaded_file)
        
# # #         # Convert image to RGB if it's not already
# # #         if image.mode != "RGB":
# # #             image = image.convert("RGB")
        
# # #         # Display the uploaded image
# # #         st.image(image, caption='Uploaded Image.', use_column_width=True)
# # #         st.write("")
# # #         st.write("Classifying...")

# # #         # Preprocess the image for prediction
# # #         img = image.resize((150, 150))  # Resize the image to the desired dimensions
# # #         img_array = np.array(img)  # Convert the image to an array
# # #         img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch
# # #         img_array = img_array / 255.0  # Normalize the image data

# # #         # Make prediction
# # #         prediction = model.predict(img_array)
# # #         predicted_class_index = np.argmax(prediction)
# # #         confidence = prediction[0][predicted_class_index]
# # #         confidence_threshold = 0.3  # Lowering the threshold for confidence

# # #         class_list = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']

# # #         if confidence >= confidence_threshold:
# # #             predicted_class = class_list[predicted_class_index]
# # #             # Display the predicted class
# # #             st.write(f"Predicted class: {predicted_class}")
# # #         else:
# # #             # Display confidences for all classes in the error message
# # #             confidence_str = ", ".join([f"{class_list[i]}: {pred:.2f}" for i, pred in enumerate(prediction[0])])
# # #             st.write(f"Gambar tidak valid untuk pendeteksian penyakit tanaman semangka. Silakan coba dengan gambar lain. Confidences: {confidence_str}")

# # #     except Exception as e:
# # #         st.write(f"Error: {e}")

# # # else:
# # #     st.write("Silakan unggah gambar.")
# # import streamlit as st
# # import numpy as np
# # from PIL import Image
# # import tensorflow as tf

# # # Load trained model
# # model = tf.keras.models.load_model('leaf_disease_classifier.h5')

# # # Streamlit App
# # st.title("Sistem Pendeteksi Penyakit Tanaman Semangka")

# # uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     try:
# #         # Load the image
# #         image = Image.open(uploaded_file)
        
# #         # Convert image to RGB if it's not already
# #         if image.mode != "RGB":
# #             image = image.convert("RGB")
        
# #         # Display the uploaded image
# #         st.image(image, caption='Uploaded Image.', use_column_width=True)
# #         st.write("")
# #         st.write("Classifying...")

# #         # Preprocess the image for prediction
# #         img = image.resize((150, 150))  # Resize the image to the desired dimensions
# #         img_array = np.array(img)  # Convert the image to an array
# #         img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch
# #         img_array = img_array / 255.0  # Normalize the image data

# #         # Make prediction
# #         prediction = model.predict(img_array)
# #         predicted_class_index = np.argmax(prediction)
# #         confidence = prediction[0][predicted_class_index]
# #         confidence_threshold = 0.5  # Threshold for classifying as valid class
# #         high_confidence_threshold = 0.7  # Higher threshold to ensure the image is valid

# #         class_list = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']

# #         # Display confidences for all classes for debugging
# #         st.write("Predictions:")
# #         for idx, class_name in enumerate(class_list):
# #             st.write(f"{class_name}: {prediction[0][idx]:.2f}")

# #         if confidence >= high_confidence_threshold:
# #             predicted_class = class_list[predicted_class_index]
# #             # Display the predicted class
# #             st.write(f"Predicted class: {predicted_class}")
# #         else:
# #             st.write("Gambar tidak valid untuk pendeteksian penyakit tanaman semangka. Silakan coba dengan gambar lain.")

# #     except Exception as e:
# #         st.write(f"Error: {e}")

# # else:
# #     st.write("Silakan unggah gambar.")
# import streamlit as st
# import numpy as np
# from PIL import Image
# import tensorflow as tf

# # Load trained model
# model = tf.keras.models.load_model('leaf_disease_classifier.h5')

# # Streamlit App
# st.title("Sistem Pendeteksi Penyakit Tanaman Semangka")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         # Load the image
#         image = Image.open(uploaded_file)
        
#         # Display the uploaded image
#         st.image(image, caption='Uploaded Image.', use_column_width=True)
#         st.write("")
#         st.write("Classifying...")

#         # Preprocess the image for prediction
#         img = image.resize((150, 150))  # Resize the image to the desired dimensions
#         img_array = np.array(img)  # Convert the image to an array
#         img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch
#         img_array = img_array / 255.0  # Normalize the image data

#         # Make prediction
#         prediction = model.predict(img_array)
#         predicted_class_index = np.argmax(prediction)
#         class_list = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']
#         predicted_class = class_list[predicted_class_index]

#         # Display the predicted class
#         st.write(f"Predicted class: {predicted_class}")

#     except Exception as e:
#         st.write("Gambar tidak valid untuk pendeteksian penyakit tanaman semangka. Silakan coba dengan gambar lain.")

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('leaf_disease_classifier2.h5')

# Streamlit App
st.title("Sistem Pendeteksi Penyakit Tanaman Semangka")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the image
        image = Image.open(uploaded_file)
        # Display the uploaded image
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for prediction
        img = image.resize((150, 150))  # Resize the image to the desired dimensions
        img_array = np.array(img)  # Convert the image to an array
        img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch
        img_array = img_array / 255.0  # Normalize the image data

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        class_list = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']
        predicted_class = class_list[predicted_class_index]

        # Display the predicted class
        st.write(f"Predicted class: {predicted_class}")

    except Exception as e:
        st.error("Gambar tidak valid. Silakan unggah gambar lain.")
