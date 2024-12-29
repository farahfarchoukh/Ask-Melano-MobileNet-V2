# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

# TensorFlow GPU memory configuration
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Define a Flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH_MOBILENET = 'path/to/your/model_MobileNet_Melanoma.h5'

# Load your trained MobileNet model
model_mobilenet = load_model(MODEL_PATH_MOBILENET)

# Folder for uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_predict_mobilenet(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)

    class_labels_mobilenet = [ 
        "actinic keratosis",
        "basal cell carcinoma",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "pigmented benign keratosis",
        "seborrheic keratosis",
        "squamous cell carcinoma",
        "vascular lesion"
    
        # Add your class labels here
    ]

    result = class_labels_mobilenet[preds[0]]
    return result

@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('index.html')

@app.route('/predict_mobilenet', methods=['POST'])
def upload_mobilenet():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        preds_mobilenet = model_predict_mobilenet(file_path, model_mobilenet)
        return render_template('index.html', prediction=preds_mobilenet)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
