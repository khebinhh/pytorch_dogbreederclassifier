import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from model import DogBreedClassifier, save_checkpoint
from data_loader import DogDataset

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(num_epochs=20, batch_size=1, learning_rate=0.0005):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    # Initialize dataset and loaders
    print("\n=== Starting Training Process ===", flush=True)
    print("Initializing dataset...", flush=True)
    dataset = DogDataset()
    train_loader, val_loader, test_loader, classes = dataset.get_data_loaders(batch_size)
    print(f"Dataset loaded successfully. Number of classes: {len(classes)}", flush=True)
    print(f"Available classes: {', '.join(classes)}", flush=True)
    print("\nDataset Statistics:", flush=True)
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}", flush=True)
    
    # Initialize model
    print("\nInitializing model...", flush=True)
    model = DogBreedClassifier(len(classes)).to(device)
    print("Model initialized successfully.", flush=True)
    print("\nStarting training loop...", flush=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Warmup scheduler
    def warmup_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    early_stopping = EarlyStopping()
    
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        # Gradually unfreeze layers every 8 epochs after epoch 10
        if epoch >= 10 and (epoch - 10) % 8 == 0:
            layer_to_unfreeze = f'layer{3 - (epoch-10)//8}'  # Unfreeze from layer3 to layer1
            if hasattr(model.model, layer_to_unfreeze):
                model.unfreeze_layer(layer_to_unfreeze)
                print(f"Unfreezing {layer_to_unfreeze}")
        
        model.train()
        running_loss = 0.0
        
        # Training loop
        print(f"\nEpoch {epoch+1}/{num_epochs}", flush=True)
        print("Training phase...", flush=True)
        progress_bar = tqdm(train_loader, desc=f'Training', leave=True, position=0)
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update warmup scheduler
            if epoch < 5:  # During warmup period
                warmup_scheduler.step()
            
            running_loss += loss.item()
            avg_loss = running_loss/(batch_idx + 1)
            # Calculate accuracy for current batch
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / labels.size(0)
            
            progress_bar.set_postfix({
                'batch': f'{batch_idx+1}/{len(train_loader)}',
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'correct': f'{correct}/{labels.size(0)}'
            }, refresh=True)
            
            # Print detailed metrics every batch for small dataset
            print(f"\nEpoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_loader)}]")
            print(f"Training Loss: {avg_loss:.4f}")
            print(f"Batch Accuracy: {accuracy:.2f}% ({correct}/{labels.size(0)} correct)")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save intermediate checkpoint every 2 batches
            if (batch_idx + 1) % 2 == 0:
                save_checkpoint(
                    model, optimizer, scheduler, 
                    epoch, avg_loss, accuracy,
                    'intermediate_model.pth'
                )
            
        # Validation loop
        print("\nValidation phase...", flush=True)
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation', leave=True, position=0)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f'\nValidation Results - Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}% ({correct}/{total} correct)')
        print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss)
        
        # Save checkpoint if best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, accuracy, 'best_model.pth')
            
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    return model, classes

if __name__ == "__main__":
    train_model()
