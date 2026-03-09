import os
import datetime
import shutil
import csv
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from loss import BinaryFocalLoss
from tqdm import tqdm

# Import our custom modules
from dataloader import get_dataloader, get_mixed_dataloader
from model import FightDetectionModel, CNNGRU, ViolenceX3D

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(config_path="config.yaml"):
    # ==========================================
    # 1. Parse Configuration
    # ==========================================
    config = load_config(config_path)
    
    TRAIN_DIR = config['train_dirs']
    use_ntu_data = False
    if "NTU_CCTV" in TRAIN_DIR[-1]:
        NTU_CCTV_DIR = TRAIN_DIR.pop(len(TRAIN_DIR) - 1) 
        use_ntu_data = True
    VAL_DIR = config['val_dir']
    NUM_FRAMES = config['num_frames']
    BATCH_SIZE = config['batch_size']
    LEARNING_RATE = float(config['learning_rate'])
    EPOCHS = config['epochs']
    NUM_CLASSES = config['num_classes']
    NETWORK_NAME = config['network_name']
    LOG_GRAD_NORM = config.get('log_grad_norm', False)
    SIZE = config['image_size']
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Initialization & TensorBoard Setup
    # ==========================================
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'fight_detection_{NETWORK_NAME}_{current_time}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save a copy of the config inside the log directory for reproducibility
    shutil.copy(config_path, os.path.join(log_dir, "config.yaml"))
    
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard initialized. Logging to: {log_dir}")

    # Load Dataloaders
    print("Initializing dataloaders...")
    if not use_ntu_data:
        train_loader, val_loader = get_dataloader(
            train_path=TRAIN_DIR, 
            val_path=VAL_DIR, 
            num_frames=NUM_FRAMES, 
            batch_size=BATCH_SIZE,
            size=SIZE
        )
    else:
        train_loader, val_loader = get_mixed_dataloader(
            regular_train_dirs=TRAIN_DIR,
            ntu_train_dir=NTU_CCTV_DIR,
            val_dir=VAL_DIR,
            ntu_json_path=os.path.join(NTU_CCTV_DIR, "groundtruth.json"),
            num_frames=NUM_FRAMES, 
            batch_size=BATCH_SIZE,
            size=SIZE
        )
    print("Total samples in train_loader:", len(train_loader.dataset))
    print("Total samples in val_loader:", len(val_loader.dataset))

    # Initialize Model dynamically based on YAML config
    print(f"Initializing model: {NETWORK_NAME}")
    if NETWORK_NAME == "FightDetectionModel":
        model = FightDetectionModel(num_classes=NUM_CLASSES, lstm_hidden_size=256, use_pretrained=True)
    elif NETWORK_NAME == "CNNGRU":
        model = CNNGRU(num_classes=NUM_CLASSES, cnn_out_features=512, rnn_hidden_size=256, num_layers=2)
    elif NETWORK_NAME == "ViolenceX3D":
        model = ViolenceX3D(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown network name: {NETWORK_NAME}")
        
    model = model.to(device)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Setup Loss Function based on NUM_CLASSES
    if NUM_CLASSES == 1:
        # We use BCEWithLogitsLoss for numerical stability. Ensure your model outputs raw logits!
        losses = {"BinaryFocalLoss": BinaryFocalLoss,
                  "BCEWithLogitsLoss": nn.BCEWithLogitsLoss}
        criterion = losses[config['loss_type']]()
        print(f"Using {config['loss_type']}.")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Using CrossEntropyLoss ({NUM_CLASSES} Classes).")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_acc = 0.0
    global_step = 0
    best_model_path = os.path.join(log_dir, "best_model.pth")

    # ==========================================
    # 3. Training Loop
    # ==========================================
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in loop:
            # Note: For X3D, you may need inputs = inputs.permute(0, 2, 1, 3, 4).to(device)
            inputs = inputs.to(device)
            
            # Format labels based on Loss Function
            if NUM_CLASSES == 1:
                labels_target = labels.float().unsqueeze(1).to(device)
            else:
                labels_target = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels_target)
            loss.backward()

            # Log Gradient Norm if enabled
            if LOG_GRAD_NORM:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Gradient/Norm', total_norm, global_step)

            optimizer.step()

            # Prediction Logic
            if NUM_CLASSES == 1:
                # Apply sigmoid and threshold at 0.5
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
            else:
                # Get index of max logit
                _, predicted = torch.max(outputs.data, 1)
            
            train_loss += loss.item() * inputs.size(0)
            train_total += labels_target.size(0)
            train_correct += (predicted == labels_target).sum().item()

            loop.set_postfix(loss=loss.item())
            writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
            global_step += 1

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100 * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                inputs = inputs.to(device)
                
                if NUM_CLASSES == 1:
                    labels_target = labels.float().unsqueeze(1).to(device)
                else:
                    labels_target = labels.long().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels_target)

                if NUM_CLASSES == 1:
                    predicted = (torch.sigmoid(outputs) >= 0.5).float()
                else:
                    _, predicted = torch.max(outputs.data, 1)

                val_loss += loss.item() * inputs.size(0)
                val_total += labels_target.size(0)
                val_correct += (predicted == labels_target).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100 * val_correct / val_total

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        writer.add_scalar('Loss/Train_Epoch', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"--> Validation accuracy improved. Saving model...")
            torch.save(model.state_dict(), best_model_path)

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    writer.close()

    # ==========================================
    # 4. Final Inference / Evaluation
    # ==========================================
    print("\nStarting final inference on the validation set...")
    
    # Load the best weights
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Final Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if NUM_CLASSES == 1:
                predicted = (torch.sigmoid(outputs) >= 0.5).int().view(-1)
            else:
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.int()
            
            # Move to CPU and convert to lists
            labels_cpu = labels.int().cpu().tolist()
            preds_cpu = predicted.cpu().tolist()
            
            for true_lbl, pred_lbl in zip(labels_cpu, preds_cpu):
                results.append({"true_label": true_lbl, "predicted_label": pred_lbl})

    # Map results to video paths
    # Note: This relies on val_loader having shuffle=False so the order matches val_loader.dataset
    if hasattr(val_loader.dataset, 'samples'):
        inference_file = os.path.join(log_dir, "inference_results.csv")
        with open(inference_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Video Path", "True Label", "Predicted Label", "Correct"])
            
            for idx, res in enumerate(results):
                # Using assuming samples stores (path, label)
                video_path = val_loader.dataset.samples[idx]
                is_correct = (res["true_label"] == res["predicted_label"])
                writer.writerow([video_path, res["true_label"], res["predicted_label"], is_correct])
                
        print(f"Inference complete. Results saved to: {inference_file}")
    else:
        print("Warning: Could not access dataset.samples to map video paths. Check your dataloader structure.")

if __name__ == "__main__":
    train_model("config.yaml")