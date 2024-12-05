import torch
import numpy as np
from PIL import Image

def predict_breed(model, image_tensor, classes, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, 5)
        
        results = []
        for i in range(5):
            breed = classes[top_class[0][i]]
            prob = top_prob[0][i].item() * 100
            results.append((breed, prob))
            
    return results

def format_breed_name(breed_name):
    # Convert from directory format to readable format
    return breed_name.replace('_', ' ').title()
