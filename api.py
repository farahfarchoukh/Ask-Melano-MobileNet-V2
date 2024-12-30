import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Define the Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'path/to/your/model.pth'  # Update this path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNetV2Wrapper(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV2Wrapper, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 1)  # For binary classification

    def forward(self, image):
        x = self.model.features(image)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.size(0), -1)
        out = self.model.classifier(x)
        return out

# Initialize the model and load weights
model = MobileNetV2Wrapper(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Set the model to evaluation mode

# Folder for uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def model_predict(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.no_grad():
        preds = model(img)
        preds = torch.sigmoid(preds).cpu().numpy()  # Apply sigmoid for binary classification
    return preds[0][0]  # Return the prediction

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        prediction = model_predict(file_path)
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
