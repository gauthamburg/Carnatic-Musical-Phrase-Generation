
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe", pe.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerVAE(nn.Module):
    def __init__(
        self,
        input_dim=1,
        seq_len=128,
        d_model=64,
        nhead=4,
        num_layers=2,
        latent_dim=2
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # Input embedding
        self.input_embedding = nn.Linear(
            input_dim, d_model
        )

        self.positional_encoding = PositionalEncoding(
            d_model, seq_len
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Latent space
        self.mu_layer = nn.Linear(
            seq_len * d_model, latent_dim
        )

        self.logvar_layer = nn.Linear(
            seq_len * d_model, latent_dim
        )

        # Project latent vector back to sequence
        self.latent_projection = nn.Linear(
            latent_dim, seq_len * d_model
        )

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(
            d_model, input_dim
        )

    def encode(self, x):
        x = self.input_embedding(x)
        x = self.positional_encoding(x)

        encoded = self.encoder(x)

        encoded = encoded.reshape(
            encoded.size(0), -1
        )

        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):
        x = self.latent_projection(z)

        x = x.reshape(
            z.size(0), self.seq_len, self.d_model
        )

        x = self.positional_encoding(x)
        x = self.decoder(x)

        return self.output_layer(x)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        reconstructed = self.decode(z)

        return reconstructed, mu, logvar


def vae_loss(x, reconstructed, mu, logvar):
    reconstruction_loss = F.mse_loss(
        reconstructed, x, reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + kl_loss
