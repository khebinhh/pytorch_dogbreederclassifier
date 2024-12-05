import torch
import torch.nn as nn
from torchvision import models

class DogBreedClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedClassifier, self).__init__()
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Initially freeze all layers except the last block and fc layer
        self.freeze_layers()
            
        # Modify the final fully connected layer with layer normalization
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize the new layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def freeze_layers(self, unfreeze_layer4=True):
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze the final block and fc layer
        if unfreeze_layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        
        # Always unfreeze the fc layer
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_layer(self, layer_name):
        # Gradually unfreeze layers during training
        if hasattr(self.model, layer_name):
            for param in getattr(self.model, layer_name).parameters():
                param.requires_grad = True
            
    def forward(self, x):
        return self.model(x)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'accuracy': accuracy
    }, filepath)

def load_checkpoint(model, optimizer, scheduler, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
