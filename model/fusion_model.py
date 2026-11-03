import torch
import torch.nn as nn
import torch.nn.functional as F
  
class FusionModel(nn.Module):
      '''
      Late fusion MLP.
      Input:  11 features (3 CV + 4 behaviour + 2 quiz + 2 reserved compatibility slots)
      Output: 3 classes — LOW(0), MEDIUM(1), OVERLOADED(2)
      '''
      def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(11, 64)
          self.fc2 = nn.Linear(64, 32)
          self.fc3 = nn.Linear(32, 16)
          self.out = nn.Linear(16, 3)
          self.dropout = nn.Dropout(0.3)
  
      def forward(self, x):
          x = F.relu(self.fc1(x))
          x = self.dropout(x)
          x = F.relu(self.fc2(x))
          x = self.dropout(x)
          x = F.relu(self.fc3(x))
          return self.out(x)   
