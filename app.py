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
       
