import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image

class DogDataset:
    def __init__(self, data_dir='stanford-dogs-dataset'):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_data_loaders(self, batch_size=32):
        dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'Images'),
            transform=self.train_transform
        )
        
        # Split dataset into train, validation, and test (adjusted for small dataset)
        total_size = len(dataset)
        train_size = max(2, int(0.6 * total_size))
        val_size = max(1, int(0.2 * total_size))
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Apply different transforms
        val_dataset.dataset.transform = self.transform
        test_dataset.dataset.transform = self.transform
        
        # Reduced num_workers to 2 as requested
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader, dataset.classes

    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
