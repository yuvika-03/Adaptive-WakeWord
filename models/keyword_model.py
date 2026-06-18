import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class KeywordCNN(nn.Module):
    def __init__(self, n_mels=40, num_classes=12):
        super(KeywordCNN, self).__init__()
        
        self.n_mels = n_mels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d((2, 2))
        
        # Calculate adaptive pooling size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, n_mels, time_steps)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

class KeywordDetector:
    def __init__(self, model_path, config):
        self.config = config
        # Use the number of classes from config
        num_classes = getattr(config, 'NUM_KEYWORD_CLASSES', 12)
        self.model = KeywordCNN(n_mels=config.N_MELS, num_classes=num_classes)
        # Load model if it exists
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
    def detect(self, features):
        """Detect keyword in features"""
        with torch.no_grad():
            # Ensure features is a torch tensor
            if not isinstance(features, torch.Tensor):
                features = torch.FloatTensor(features)
            
            # Add batch and channel dimensions if needed
            if len(features.shape) == 2:
                features = features.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            elif len(features.shape) == 3:
                features = features.unsqueeze(0)  # Add batch dim
            
            # Make sure the tensor is on the right device
            if next(self.model.parameters()).is_cuda:
                features = features.cuda()
            
            output = self.model(features)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            max_prob, pred_class = torch.max(probabilities, dim=1)
            
        return pred_class.item(), max_prob.item()