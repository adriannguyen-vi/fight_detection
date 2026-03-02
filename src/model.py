import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SelfAttention(nn.Module):
    """
    Self-attention layer as described in the paper to focus on significant frames.
    """
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        # hidden_size * 2 because of the Bidirectional LSTM
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_size * 2)
        
        # Calculate attention weights for each frame in the sequence
        attn_weights = F.softmax(self.attention(lstm_outputs), dim=1)
        
        # Multiply weights by LSTM outputs and sum across the time dimension
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        
        return context_vector, attn_weights

class FightDetectionModel(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=256, use_pretrained=True):
        super(FightDetectionModel, self).__init__()
        
        # 1. Feature Extraction: Xception
        # num_classes=0 strips the final classification layer, returning raw pooled features
        self.feature_extractor = timm.create_model(
            'xception', 
            pretrained=use_pretrained, 
            num_classes=0 
        )
        # Xception outputs 2048-dimensional feature vectors
        feature_dim = self.feature_extractor.num_features 

        # 2. Temporal Modeling: Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # 3. Attention Layer
        self.attention = SelfAttention(lstm_hidden_size)

        # 4. Classification Head: 3 Dense Layers (1024, 50, 2)
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 1024)
        self.fc2 = nn.Linear(1024, 50)
        self.fc3 = nn.Linear(50, num_classes)

        self.act = F.sigmoid if num_classes == 1 else F.softmax
    def forward(self, x):
        # Input shape expected: (batch_size, seq_len, channels, height, width)
        # Note: If your dataloader outputs (batch_size, channels, seq_len, h, w), 
        # you must permute it before passing it here: x = x.permute(0, 2, 1, 3, 4)
        batch_size, seq_len, c, h, w = x.size()
        # print(f"x.shape: {x.shape}")

        # Reshape to process all frames from all batches through the CNN simultaneously
        x = x.reshape(batch_size * seq_len, c, h, w)

        # Extract spatial features using Xception
        features = self.feature_extractor(x)

        # Reshape back to sequence format for the LSTM
        features = features.view(batch_size, seq_len, -1)

        # Pass through Bi-LSTM
        lstm_out, _ = self.lstm(features)

        # Apply Self-Attention
        context, _ = self.attention(lstm_out)

        # Pass through final dense layers with specified activations
        out = F.relu(self.fc1(context))
        out = F.relu(self.fc2(out))
        # out = self.act(self.fc3(out))
        out = self.fc3(out)

        return out

class CNNGRU(nn.Module):
    def __init__(self, num_classes=2, cnn_out_features=512, rnn_hidden_size=256, num_layers=2):
        super(CNNGRU, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, cnn_out_features)
        )
        self.rnn = nn.GRU(cnn_out_features, rnn_hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        batch_size, frames, channels, height, width = x.size()
        # print(x.shape)
        c_in = x.reshape(batch_size * frames, channels, height, width)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, frames, -1)
        r_out, h_n = self.rnn(r_in)
        out = self.fc(r_out[:, -1, :])
        return out

class X3D(nn.Module):
    def __init__(self, num_classes=2, model_version="x3d_m", pretrained=True):

        self.model = torch.hub.load('facebookresearch/pytorchvideo', model_version, pretrained=pretrained)
        in_features = self.model.blocks.proj.in_features
        self.model.blocks.proj = nn.Linear(in_features, num_classes)

    def forward(self, x):
        print(f"In X3D, x.shape:",x.shape)
        x = self.model(x)
        return x
    