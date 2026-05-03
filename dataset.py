# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import config

class PneumoniaImgDataset(Dataset):
    def __init__(self, img_paths, labels, is_train=True):
        self.img_paths = img_paths
        self.labels = labels
        
        # Using variables from config.py
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((config.resize_x, config.resize_y)),
                transforms.Grayscale(num_output_channels=config.input_channels),
                transforms.RandomAffine(degrees=3, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config.resize_x, config.resize_y)),
                transforms.Grayscale(num_output_channels=config.input_channels),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = self.transform(img)
        # Convert label to float32 tensor for BCEWithLogitsLoss
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, label

def get_dataloader(dataset, is_train=True):
    return DataLoader(dataset, batch_size=config.batchsize, shuffle=is_train)