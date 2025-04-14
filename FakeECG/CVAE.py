import torch
from torch import nn
import torch.nn.functional as F

"""
FLUJO:

1. Embeddings condicionales:

La etiqueta de clase se convierte en una secuencia con self.embed_class(labels).
La señal ECG se procesa con self.embed_data(x).
Se concatenan ambos (torch.cat), creando una entrada combinada.

2. Codificación:
self.encode(x) genera mu y log_var.

3. Reparametrización:
Se obtiene z = self.reparameterize(mu, log_var).

4. Concatenación con la etiqueta:
z se combina con la condición (labels).

5. Decodificación:
Se pasa por el decoder para reconstruir la señal ECG.

6. Función de Pérdida
El modelo usa una pérdida que combina:
    Reconstrucción (MSE):
    Compara la señal reconstruida con la original.
    Se usa F.mse_loss(recons, x, reduction='sum').

    Divergencia KL:
    Regulariza la distribución latente para que sea lo más parecida posible a una gaussiana estándar (N(0,1)).

    Se calcula como:
    KLD = −0.5∑(1+log𝜎^2−𝜇^2−𝜎^2)

    Se pondera con beta para ajustar la regularización.

"""

class ConditionalVAE(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, latent_dim=16, hidden_dims=None, seq_length=1000):
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.seq_length = seq_length  # Longitud de la señal ECG

        # Embedding de etiquetas condicionales
        self.embed_class = nn.Linear(num_classes, seq_length) # Convierte la etiqueta de clase en una representación del mismo tamaño que la señal ECG.
        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1) # procesar la señal original antes de concatenarla con la etiqueta embebida.

        # Definir capas ocultas si no están especificadas
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        in_channels += 1  # Para incluir la etiqueta embebida
        modules = []
        
        # **Encoder (Conv1d)** 
        # transforma la entrada (señal ECG + condición) en una representación latente.
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(h_dim), # estabilizar entrenamiento
                nn.LeakyReLU()         # funcion de activacion 
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * (seq_length // 8), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * (seq_length // 8), latent_dim)


        # **Decoder (ConvTranspose1d)**
        # recibe un vector latente (z) y lo transforma nuevamente en una señal ECG:
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * (seq_length // 8))
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
            #nn.Tanh()
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

    # reparametrizacion de Kingma & Welling (2014)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, labels):
        labels = labels.float()
        embedded_class = self.embed_class(labels).unsqueeze(1)  # Expandir para concatenar con ECG
        embedded_input = self.embed_data(x)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, labels], dim=1)

        return self.decode(z), x, mu, log_var

    def loss_function(self, recons, x, mu, log_var, beta=1):
        recon_loss = F.mse_loss(recons, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kld_loss
