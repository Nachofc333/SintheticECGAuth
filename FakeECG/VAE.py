import torch
import torch.nn as nn
import torch.nn.functional as F

class EstandarVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16, hidden_dims=None, seq_length=256):
        super(EstandarVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.seq_length = seq_length

        # Definir capas ocultas si no están especificadas
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        modules = []
        
        # **Encoder (Conv1d)**
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * (seq_length // 8), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * (seq_length // 8), latent_dim)

        # **Decoder (ConvTranspose1d)**
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * (seq_length // 8))
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 128, self.seq_length // 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), x, mu, log_var

    def loss_function(self, recons, x, mu, log_var, beta=1):
        recon_loss = F.mse_loss(recons, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kld_loss
