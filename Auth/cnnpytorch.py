import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiresBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(MultiresBlock, self).__init__()
        
        self.shortcut = nn.Conv1d(in_channels, n_filters * (1+2+4), kernel_size=1, padding='same')
        
        self.conv3x3 = nn.Conv1d(in_channels, n_filters, kernel_size=15, padding='same')
        self.conv5x5 = nn.Conv1d(n_filters, n_filters*2, kernel_size=15, padding='same')
        self.conv7x7 = nn.Conv1d(n_filters*2, n_filters*4, kernel_size=15, padding='same')
        
        self.batch_norm = nn.BatchNorm1d(n_filters * (1+2+4))
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        shortcut = F.relu(self.batch_norm(shortcut))
        
        conv3x3 = F.relu(self.conv3x3(x))
        conv5x5 = F.relu(self.conv5x5(conv3x3))
        conv7x7 = F.relu(self.conv7x7(conv5x5))
        
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        out = self.batch_norm(out)
        out = F.relu(out + shortcut)
        
        return out

class SPPLayer(nn.Module):
    def __init__(self, spp_windows):
        super(SPPLayer, self).__init__()
        self.poolings = nn.ModuleList([nn.AdaptiveMaxPool1d(out_size) for out_size in spp_windows])
        
    def forward(self, x):
        pooled = [pool(x).view(x.shape[0], -1) for pool in self.poolings]
        return torch.cat(pooled, dim=1)

class CNNModel(nn.Module):
    def __init__(self, seq_len, n_classes, dp_rate=0.25):
        super(CNNModel, self).__init__()
        
        self.multires_block = MultiresBlock(1, 32)
        self.spp_layer = SPPLayer([8, 16, 32])
        
        self.fc1 = nn.Linear(12544, 128)

        self.batch_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dp_rate)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        x = self.multires_block(x)
        x = self.spp_layer(x)
        x = torch.flatten(x, start_dim=1)  # Aplanar salida
        x = F.relu(self.batch_norm(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# Para crear el modelo:
# model = CNNModel(seq_len=W_LEN, n_classes=90)
