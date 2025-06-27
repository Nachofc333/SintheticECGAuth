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
    def __init__(self, in_channels=1, num_classes=10, latent_dim=16, hidden_dims=None, seq_length=256):
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
        # recibe un vector latente y lo transforma nuevamente en una señal ECG:
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

"""
Modelo 1:

class ConditionalVAE(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, latent_dim=16, hidden_dims=None, seq_length=256):
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.seq_length = seq_length

        self.embed_class = nn.Linear(num_classes, seq_length)
        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        in_channels += 1
        modules = []

        # Encoder
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

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * (seq_length // 8))
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ))

        # NUEVA CAPA 1 (más profunda)
        modules.append(nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU()
        ))

        # NUEVA CAPA 2 (más profunda)
        modules.append(nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU()
        ))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Tanh()
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

    def forward(self, x, labels):
        labels = labels.float()
        embedded_class = self.embed_class(labels).unsqueeze(1)
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

Modelo 2:

class ConditionalVAE(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, latent_dim=32, hidden_dims=None, seq_length=256):
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.seq_length = seq_length

        # 🔹 Mini red para embedding de clase
        self.embed_class = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.LeakyReLU(),
            nn.Linear(64, seq_length),
        )
        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        # 🔹 Definir dimensiones ocultas +1 capa
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        in_channels += 1  # incluir etiqueta embebida
        modules = []

        # 🔹 Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * (seq_length // 2**len(hidden_dims)), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * (seq_length // 2**len(hidden_dims)), latent_dim)

        # 🔹 Decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * (seq_length // 2**len(hidden_dims)))
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ))

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 256, self.seq_length // 2**4)  # 256 = hidden_dims[0]
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x, labels):
        labels = labels.float()
        embedded_class = self.embed_class(labels).unsqueeze(1)  # (B, 1, seq_length)
        embedded_input = self.embed_data(x)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, labels], dim=1)
        return self.decode(z), x, mu, log_var

    def fft_loss(self, recons, x):
        # Transformada de Fourier
        fft_recons = torch.fft.rfft(recons, dim=-1)
        fft_x = torch.fft.rfft(x, dim=-1)
        return F.mse_loss(torch.abs(fft_recons), torch.abs(fft_x), reduction='sum')

    def loss_function(self, recons, x, mu, log_var, beta=1, fft_weight=0.1):
        recon_loss = F.mse_loss(recons, x, reduction='sum')
        fft_loss = self.fft_loss(recons, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + beta * kld_loss + fft_weight * fft_loss
        return total_loss


Modelo3:

class ConditionalVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10,
                 latent_dim: int = 16,
                 hidden_dims: list[int] | None = None,
                 seq_length: int = 256):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_classes = num_classes

        # --- 1. Embedding de la clase ----
        self.class_embedding = nn.Embedding(num_classes, seq_length)
        self.class_embed_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

        # (opcional) proyección del canal de datos
        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        # --- 2. Definir dimensiones ocultas ---
        if hidden_dims is None:
            # +256 como cuarta capa
            hidden_dims = [32, 64, 128, 256]

        self.hidden_dims = hidden_dims.copy()          # guardamos para el decoder
        self.downsample_factor = 2 ** len(hidden_dims) # p. ej. 2⁴ = 16

        # --- 3. Encoder ---
        modules = []
        in_channels += 1  # +1 por el canal de la clase embebida

        current_len = seq_length
        for h_dim in hidden_dims:
            current_len //= 2  # cada conv stride-2
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm([h_dim, current_len]),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        flattened_dim = hidden_dims[-1] * (seq_length // self.downsample_factor)
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_var = nn.Linear(flattened_dim, latent_dim)

        # --- 4. Decoder ---
        self.decoder_input = nn.Linear(latent_dim + num_classes, flattened_dim)

        hidden_dims_rev = hidden_dims[::-1]            # copia invertida
        modules = []
        for i in range(len(hidden_dims_rev) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims_rev[i],
                                       hidden_dims_rev[i + 1],
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.GroupNorm(4, hidden_dims_rev[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(hidden_dims_rev[-1], 1,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1)
            # nn.Sigmoid() o nn.Tanh() según rango deseado
        )

    # ---------- Métodos utilitarios ----------
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(
            -1,
            self.hidden_dims[-1],                  # 256
            self.seq_length // self.downsample_factor  # 256 // 16 = 16
        )
        result = self.decoder(result)
        result = self.final_layer(result)

        # volvemos exactamente al tamaño de entrada original
        result = F.interpolate(
            result, size=self.seq_length,
            mode='linear', align_corners=False
        )
        return result

    # ---------- Forward ----------
    def forward(self, x, labels):
        labels_idx = (labels.argmax(dim=1)
                      if labels.ndim > 1 else labels)
        class_embedded = self.class_embedding(labels_idx).unsqueeze(1)  # (B,1,L)
        class_embedded = self.class_embed_conv(class_embedded)

        x_embed = self.embed_data(x)
        x = torch.cat([x_embed, class_embedded], dim=1)

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        # concatenamos la etiqueta una vez en el espacio latente
        z = torch.cat([z, labels], dim=1)

        x_recon = self.decode(z)
        return x_recon, x, mu, log_var

    def forward(self, x, labels):
        labels_idx = labels.argmax(dim=1) if labels.ndim > 1 else labels
        class_embedded = self.class_embedding(labels_idx).unsqueeze(1)  # (B, 1, L)
        class_embedded = self.class_embed_conv(class_embedded)

        x_embed = self.embed_data(x)
        x = torch.cat([x_embed, class_embedded], dim=1)

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, labels], dim=1)
        x_recon = self.decode(z)

        return x_recon, x, mu, log_var
    
    def loss_function(recon_x, x, mu, logvar):
        recon_loss = F.smooth_l1_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss

Modelo 4:

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ConditionalVAE(nn.Module):
    def __init__(self, in_channels=1, num_classes=90, latent_dim=32, hidden_dims=None, seq_length=256):
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.seq_length = seq_length

        # Embedding de etiquetas condicionales (one-hot → proyección)
        self.embed_class = nn.Linear(num_classes, seq_length)  # (batch, seq_len)
        self.embed_data = nn.Conv1d(in_channels, in_channels, kernel_size=1)

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        in_channels += 1  # concatenamos 1 canal por la etiqueta expandida
        modules = []

        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(h_dim),
                Mish()
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * (seq_length // 8), latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * (seq_length // 8), latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * (seq_length // 8))
        hidden_dims.reverse()

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                Mish()
            ))

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(hidden_dims[-1], 1, kernel_size=3, padding=1),
            # nn.Tanh()  # si tus señales están normalizadas en [-1, 1]
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

    def forward(self, x, labels):
        labels = labels.float()  # one-hot: (batch, num_classes)
        embedded_class = self.embed_class(labels).unsqueeze(1)  # (batch, 1, seq_len)
        embedded_input = self.embed_data(x)  # (batch, 1, seq_len)
        x_cond = torch.cat([embedded_input, embedded_class], dim=1)  # (batch, 2, seq_len)

        mu, log_var = self.encode(x_cond)
        z = self.reparameterize(mu, log_var)
        z_cond = torch.cat([z, labels], dim=1)  # (batch, latent + class)
        x_recon = self.decode(z_cond)
        return x_recon, x_cond, mu, log_var

    def loss_function(self, recons, x, mu, log_var, beta=1.0, recon_type='mae+mse'):
        if recon_type == 'mse':
            recon_loss = F.mse_loss(recons, x, reduction='sum')
        elif recon_type == 'mae':
            recon_loss = F.l1_loss(recons, x, reduction='sum')
        else:
            recon_loss = 0.5 * F.mse_loss(recons, x, reduction='sum') + 0.5 * F.l1_loss(recons, x, reduction='sum')

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kld_loss


"""