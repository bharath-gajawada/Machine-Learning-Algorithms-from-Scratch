# 4.4
import torch

class PcaAutoencoder:
    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvectors = None
        self.mean = None

    def fit(self, train_loader):
        data = []
        for batch, _ in train_loader:
            batch = batch.view(batch.size(0), -1)
            data.append(batch)
        data = torch.cat(data, dim=0)

        self.mean = torch.mean(data, dim=0)
        centered_data = data - self.mean

        cov_matrix = torch.mm(centered_data.T, centered_data) / (centered_data.size(0) - 1)

        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
        
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        self.eigenvectors = eigenvectors[:, sorted_indices][:, :self.n_components]

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x_centered = x - self.mean

        encoded = torch.mm(x_centered, self.eigenvectors)
        return encoded

    def forward(self, x):
        encoded = self.encode(x)

        reconstructed = torch.mm(encoded, self.eigenvectors.T) + self.mean
        reconstructed = reconstructed.view(x.size(0), 1, 28, 28)
        return reconstructed
