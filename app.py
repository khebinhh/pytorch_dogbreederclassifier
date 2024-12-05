from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
import io
import os
from model import DogBreedClassifier, load_checkpoint
from data_loader import DogDataset
from utils import predict_breed, format_breed_name

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model and dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = DogDataset()
_, _, _, classes = dataset.get_data_loaders()
model = DogBreedClassifier(len(classes)).to(device)

# Load the trained model
if os.path.exists('best_model.pth'):
    _, _, _ = load_checkpoint(model, None, 'best_model.pth')
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = dataset.process_image(image)
        
        # Get predictions
        predictions = predict_breed(model, image_tensor, classes, device)
        
        # Format results
        results = [
            {
                'breed': format_breed_name(breed),
                'probability': f"{prob:.2f}%"
            }
            for breed, prob in predictions
        ]
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
