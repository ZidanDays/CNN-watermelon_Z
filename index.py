from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained models
modelnasnet = load_model("models/NASNetMobile.h5")
modelvgg = load_model("models/VGG16.h5")
modelxception = load_model("models/Xception.h5")
modelcnn = load_model("models/scratchCNN.h5")

# Define allowed file types for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = "uploads/" + filename
        file.save(file_path)

        # Load and preprocess the image for prediction
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image based on the model's requirements
        img_array = img_array / 255.0

        # Perform predictions using loaded models
        prediction_nasnet = modelnasnet.predict(img_array)
        prediction_vgg = modelvgg.predict(img_array)
        prediction_xception = modelxception.predict(img_array)
        prediction_cnn = modelcnn.predict(img_array)

        # Perform post-processing on predictions as needed
        # For example, convert prediction probabilities to class labels

        # Return predictions as JSON response
        return jsonify({
            'nasnet': prediction_nasnet.tolist(),
            'vgg': prediction_vgg.tolist(),
            'xception': prediction_xception.tolist(),
            'cnn': prediction_cnn.tolist()
        })
    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == "__main__":
    app.run(debug=True)
