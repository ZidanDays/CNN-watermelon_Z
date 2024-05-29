# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# tf.__version__

"""## Part 1 - Data Preprocessing

### Preprocessing the Training set
"""





from sklearn.model_selection import train_test_split
import shutil

# Direktori awal dengan gambar-gambar yang sama
original_dir = 'path/to/dataset/'

# Direktori untuk data training dan validation yang terpisah
train_dir = 'path/to/train_dataset/'
validation_dir = 'path/to/validation_dataset/'

# Membuat direktori untuk data validation jika belum ada
os.makedirs(validation_dir, exist_ok=True)

# Mendapatkan daftar gambar dalam direktori original
image_files = os.listdir(original_dir)

# Memisahkan gambar-gambar menjadi training dan validation secara acak
train_files, validation_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Memindahkan gambar-gambar ke direktori training dan validation yang sesuai
for file in validation_files:
    shutil.move(os.path.join(original_dir, file), os.path.join(validation_dir, file))








train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Mengatur proporsi data yang akan digunakan untuk validasi
)

train_generator = train_datagen.flow_from_directory(
    'path/to/dataset/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Menentukan subset sebagai data training
)

validation_generator = train_datagen.flow_from_directory(
    'path/to/dataset/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Menentukan subset sebagai data validation
)






# Directory paths for training and validation data
train_dir = 'D:/XAMPP BC/htdocs/Pendeteksi_CNN/dataset/train'
validation_dir = 'D:/XAMPP BC/htdocs/Pendeteksi_CNN/dataset/validation'


training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

"""## Part 2 - Building the CNN

### Initialising the CNN
"""

cnn = tf.keras.models.Sequential()

"""### Step 1 - Convolution"""

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

"""### Step 2 - Pooling"""

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

"""### Adding a second convolutional layer"""

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

"""### Step 3 - Flattening"""

cnn.add(tf.keras.layers.Flatten())

"""### Step 4 - Full Connection"""

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

"""### Step 5 - Output Layer"""

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""## Part 3 - Training the CNN

### Compiling the CNN
"""

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

"""### Training the CNN on the Training set and evaluating it on the Test set"""

cnn.fit(x = training_set, validation_data = test_set, epochs = 15)

"""### Saving the model"""
tf.keras.models.save_model(cnn, 'CNN_model.hdf5')
