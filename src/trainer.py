import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import our custom modules
from dataloader import get_dataloader
from model import FightDetectionModel, CNNGRU

def train_model():
    # ==========================================
    # 1. Hyperparameters & Configuration
    # ==========================================
    TRAIN_DIR = [
        "/home/adrian/fight_detection/data/RWF-2000/train",
        "/home/adrian/fight_detection/data/RWF-2000/val",
        "/home/adrian/fight_detection/data/RLVSD_Kaggle"
    ]
    VAL_DIR = "/home/adrian/fight_detection/data/fight-detection-surv-dataset" 
    
    NUM_FRAMES = 15
    BATCH_SIZE = 4     
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    NUM_CLASSES = 2 # 2 classes for CrossEntropyLoss (NonViolence=0, Violence=1)
    
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
    log_dir = os.path.join('runs', f'fight_detection_{current_time}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard initialized. Logging to: {log_dir}")

    # Load Dataloaders
    print("Initializing dataloaders...")
    train_loader, val_loader = get_dataloader(
        train_path=TRAIN_DIR, 
        val_path=VAL_DIR, 
        num_frames=NUM_FRAMES, 
        batch_size=BATCH_SIZE
    )
    print("Total samples in train_loader:", len(train_loader.dataset))
    print("Total samples in val_loader:", len(val_loader.dataset))
    # Initialize Model
    print("Initializing model...")
    # model = CNNGRU(num_classes=NUM_CLASSES, cnn_out_features=512, rnn_hidden_size=256, num_layers=2)
    model = FightDetectionModel(num_classes=NUM_CLASSES, lstm_hidden_size=256, use_pretrained=True)
    model = model.to(device)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Use Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()
    print("Using CrossEntropyLoss.")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_acc = 0.0
    global_step = 0

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
            inputs = inputs.to(device)
            
            # FIX 1: CE Loss expects labels to be type 'long' and shape (Batch_Size)
            labels_target = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels_target)
            
            loss.backward()
            optimizer.step()

            # FIX 2: CE Loss outputs shape (Batch, 2). Get the index of the max logit.
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
                
                # FIX 3: Same label formatting for validation
                labels_target = labels.long().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels_target)

                # FIX 4: Same prediction logic for validation
                _, predicted = torch.max(outputs.data, 1)

                val_loss += loss.item() * inputs.size(0)
                val_total += labels_target.size(0)
                val_correct += (predicted == labels_target).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100 * val_correct / val_total

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # 4. Write epoch metrics to TensorBoard
        writer.add_scalar('Loss/Train_Epoch', epoch_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch)

        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"--> Validation accuracy improved. Saving model...")
            torch.save(model.state_dict(), f"{log_dir}/best_fight_detection_model.pth")

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    train_model()