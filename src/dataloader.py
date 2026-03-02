import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2 # Use v2 for native video support
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ViolenceDataset(Dataset):
    def __init__(self, root_dirs, frames=10, size=229, transform=None) -> None:
        super().__init__()
        self.samples = []
        self.transform = transform
        self.frames = frames
        self.size = size
        
        # Initialize the dictionary to store cached video tensors
        self.cache = {}
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        
        for root_dir in root_dirs:
            for label, folder in enumerate(['NonViolence', 'Violence']):
                class_dir = os.path.join(root_dir, folder)
                if not os.path.exists(class_dir):
                    continue
                
                for file in os.listdir(class_dir):
                    self.samples.append((
                        os.path.join(class_dir, file),
                        label
                    ))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        
        # 1. Check if the video tensor is already in the cache
        
        frames = []
        cap = cv2.VideoCapture(path)
        while True: 
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.size, self.size))
            # H W C -> C H W
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        cap.release()

        # Handle empty corrupted videos gracefully
        if len(frames) == 0:
            # Shape is (T, C, H, W)
            video = torch.zeros((self.frames, 3, self.size, self.size))
        else:
            indices = np.linspace(0, len(frames) - 1, self.frames, dtype=int)
            sampled_frames = [frames[i] for i in indices]

            # Stack into (T, C, H, W)
            video = torch.stack(sampled_frames)
        
            # Save the raw tensor into RAM cache
            # self.cache[index] = video
        

        # 2. Apply transformations AFTER retrieving from cache
        # This guarantees random augmentations are applied freshly every epoch
        if self.transform:
            video = self.transform(video)
        
        # If your specific model (like a 2D CNN + LSTM) needs (C, T, H, W), 
        # you can permute it back HERE after the transforms:
        # video = video.permute(1, 0, 2, 3) 
        
        return video, label
        

def get_dataloader(train_path, val_path, num_frames, batch_size, size=224):
    # 1. Define Training Augmentations
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5), # Fights look the same mirrored
        v2.RandomRotation(degrees=10),  # Slight rotations simulate camera angles
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Simulates CCTV lighting changes
        # Normalization (Standard ImageNet stats - highly recommended for pre-trained CNNs)
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    # 2. Define Validation/Test Transforms (NO Augmentation, just normalization!)
    val_transform = v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Apply respective transforms to the datasets
    train_dataset = ViolenceDataset(train_path, frames=num_frames, transform=train_transform, size=size)
    val_dataset = ViolenceDataset(val_path, frames=num_frames, transform=val_transform, size=size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader