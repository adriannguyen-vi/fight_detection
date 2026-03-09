import os
import torch
import cv2
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision.transforms import v2 # Use v2 for native video support

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
class NTUCCTVDataset(Dataset):
    def __init__(self, root_dir, json_path, subset=['training'], frames=15, size=224, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.frames = frames
        self.size = size
        self.transform = transform
        self.samples = []

        with open(json_path, 'r') as f:
            data = json.load(f)
            
        database = data.get("database", {})
        subset = [x.lower() for x in subset]

        for vid_id, vid_info in database.items():
            if vid_info.get("subset", "").lower() not in  subset:
                continue

            annotations = vid_info.get("annotations", [])
            
            video_path = os.path.join(self.root_dir, f"{vid_id}.mpeg")
            if not os.path.exists(video_path):
                video_path = os.path.join(self.root_dir, f"{vid_id}.avi")
                if not os.path.exists(video_path):
                    continue

            self.samples.append({
                "path": video_path,
                "annotations": annotations,
                "frame_rate": vid_info.get("frame_rate", 30.0)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        path = sample["path"]
        annotations = sample["annotations"]
        fps = sample.get("frame_rate", 30.0)

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return torch.zeros((self.frames, 3, self.size, self.size)), 0

        # Extract all annotated fight segments (converted to frame indices)
        fight_segments = []
        for ann in annotations:
            if ann.get("label", "").lower() == "fight":
                start_f = int(ann["segment"][0] * fps)
                end_f = int(ann["segment"][1] * fps)
                fight_segments.append((start_f, end_f))

        fight_segments.sort() # Ensure they are in chronological order
        
        label = 0
        start_frame = 0
        end_frame = total_frames - 1

        if len(fight_segments) > 0:
            # Randomly decide to sample Violence (1) or Non-Violence (0)
            if random.random() > 0.5:
                # --- SAMPLE VIOLENCE ---
                # Pick a random fight segment from the list
                seg = random.choice(fight_segments)
                start_frame, end_frame = seg
                label = 1
            else:
                # --- SAMPLE NON-VIOLENCE (HARD NEGATIVE) ---
                # Calculate the "Safe Zones" (gaps between fights)
                safe_zones = []
                last_end = 0
                
                for f_start, f_end in fight_segments:
                    # If there is at least a 2-second gap, consider it a safe zone
                    if f_start - last_end > fps * 2: 
                        safe_zones.append((last_end, f_start))
                    last_end = max(last_end, f_end)
                
                # Check the gap after the last fight until the end of the video
                if total_frames - last_end > fps * 2:
                    safe_zones.append((last_end, total_frames - 1))
                
                if len(safe_zones) > 0:
                    seg = random.choice(safe_zones)
                    start_frame, end_frame = seg
                    label = 0
                else:
                    # Fallback: If the entire video is a fight with no safe zones, just return the fight
                    seg = fight_segments[0]
                    start_frame, end_frame = seg
                    label = 1

        # Bounds check to prevent OpenCV errors
        end_frame = min(end_frame, total_frames - 1)
        if start_frame >= end_frame:
            start_frame = 0
            end_frame = total_frames - 1

        # Generate evenly spaced indices within our chosen segment
        indices = np.linspace(start_frame, end_frame, self.frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.size, self.size))
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frames.append(frame)
            else:
                frames.append(torch.zeros((3, self.size, self.size)))

        cap.release()

        video = torch.stack(frames)
        if self.transform:
            video = self.transform(video)
        
        return video, label

# ==========================================
# Mixed Dataloader Function using ConcatDataset
# ==========================================
def get_mixed_dataloader(regular_train_dirs, ntu_train_dir, ntu_json_path, val_dir, num_frames, batch_size, size=224):
    """
    Combines standard folder-based datasets with the NTU JSON-based dataset.
    """
    
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5), 
        v2.RandomRotation(degrees=10),  
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    val_transform = v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Load the Standard Folder-based Datasets (e.g., RWF-2000, RLVD)
    if isinstance(regular_train_dirs, str):
        regular_train_dirs = [regular_train_dirs]
        
    standard_datasets = []
    for d in regular_train_dirs:
        standard_datasets.append(
            ViolenceDataset(d, frames=num_frames, size=size, transform=train_transform)
        )

    # 2. Load the NTU Dataset (Training subset)
    ntu_train_dataset = NTUCCTVDataset(
        root_dir=ntu_train_dir, json_path=ntu_json_path, subset=['training', 'testing','validation'], 
        frames=num_frames, size=size, transform=train_transform
    )

    # 3. Mix them together using ConcatDataset!
    combined_train_dataset = ConcatDataset(standard_datasets + [ntu_train_dataset])

    # 4. Load the Validation Dataset
    # (Assuming validation is done on your standard CCTV test folder, 
    # but you could also Concat the NTU validation subset here if you want)
    val_dataset = ViolenceDataset(val_dir, frames=num_frames, size=size, transform=val_transform)

    # 5. Create DataLoaders
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader