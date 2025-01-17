# 4.2.2

import torch
import torch.nn as nn

class CnnAutoencoder(nn.Module):
    def __init__(self, kernel_size=3, latent_dim=10):
        super(CnnAutoencoder, self).__init__()
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        padding = kernel_size // 2
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(4, 16, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.flattened_dim = 16 * 7 * 7
        self.fc1 = nn.Linear(self.flattened_dim, self.latent_dim)
        
        self.fc2 = nn.Linear(self.latent_dim, self.flattened_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, kernel_size=self.kernel_size, stride=2, padding=padding, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(1, 1, kernel_size=self.kernel_size, stride=2, padding=padding, output_padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = self.fc2(x)
        x = x.view(x.size(0), 16, 7, 7)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x