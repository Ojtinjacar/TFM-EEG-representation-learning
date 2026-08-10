import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import StandardNormalPrior

class CNNAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_channels=2, dropout=0.25):
        super().__init__()

        self.input_size = input_size  # temporal length (T)
        self.hidden_size = hidden_size
        self.n_channels = n_channels

        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hidden_size // 4, (n_channels, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # T // 2

            nn.Conv2d(hidden_size // 4, hidden_size // 2, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # T // 4

            nn.Conv2d(hidden_size // 2, hidden_size, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.Upsample(size=(1, input_size // 4), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size, hidden_size // 2, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),

            nn.Upsample(size=(1, input_size // 2), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size // 2, hidden_size // 4, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),

            nn.Upsample(size=(1, input_size), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size // 4, n_channels, (1, 1))
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x.unsqueeze(1)  # -> (B, 1, C, T)
        encoded = self.encoder(x)  # -> (B, H, 1, T//4)
        decoded = self.decoder(encoded)  # -> (B, ?, C, T)
        decoded = decoded.squeeze(2)  # -> (B, C, T)
        return decoded, encoded

    def get_embedding(self, x):
        with torch.no_grad():
            x = x.unsqueeze(1)  # -> (B, 1, C, T)
            encoded = self.encoder(x)  # -> (B, H, 1, T')
            pooled = F.adaptive_avg_pool2d(encoded, (1, 1))  # -> (B, H, 1, 1)
            return pooled.reshape(pooled.size(0), -1)


class AttentionLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_channels=2, sfreq=100,
                 lstm_hidden_size=64, lstm_layers=2, n_attention_heads=4,
                 dropout=0.25):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.input_size = input_size  # temporal length

        # ---------- Encoder ----------
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, hidden_size // 4, (n_channels, 1)),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),

            nn.Conv2d(hidden_size // 4, hidden_size // 2, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(hidden_size // 2, hidden_size, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

        self.temporal_reduction = input_size // 4  # two maxpool operations

        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.embedding_layer = nn.Linear(lstm_hidden_size * 2, hidden_size)

        # ---------- Decoder ----------
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size * 2,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.reconstruction_layer = nn.Linear(lstm_hidden_size * 2, hidden_size)

        self.decoder_deconv = nn.Sequential(
            nn.Upsample(size=(1, 625), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),

            nn.Upsample(size=(1, 1250), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size // 2, hidden_size // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),

            nn.Conv2d(hidden_size // 4, n_channels, kernel_size=(1, 1))
        )

    def encode(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.encoder_cnn(x)  # (B, H, 1, T')
        x = x.squeeze(2)         # (B, H, T')
        x = x.permute(0, 2, 1).contiguous()

        lstm_out, _ = self.encoder_lstm(x)  # (B, T', 2*LSTM_H)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        embedding = self.embedding_layer(attn_out)  # (B, T', H)
        return embedding

    def decode(self, embedding):
        x, _ = self.decoder_lstm(embedding)  # (B, T', 2*LSTM_H)
        x = self.reconstruction_layer(x)     # (B, T', H)

        x = x.permute(0, 2, 1).contiguous().unsqueeze(2)
        x = self.decoder_deconv(x)           # (B, C, ?, T)
        x = x.squeeze(2)                     # (B, C, T)
        return x

    def get_embedding(self, x):
        embedding_seq = self.encode(x)          # (B, T', hidden_dim)
        embedding = embedding_seq.mean(dim=1)   # (B, hidden_dim)
        return embedding

    def forward(self, x):
        embedding = self.encode(x)
        reconstruction = self.decode(embedding)
        return reconstruction, embedding

class MaskedAttentionLSTMAutoencoder(nn.Module):
    """
    Masked Autoencoder (MAE) for EEG signals.

    Instead of reconstructing the entire signal, the model:
    1. Masks random parts of the input signal (mask_ratio%)
    2. Processes only the visible (unmasked) parts
    3. Predicts the masked parts based on context

    This forces the model to learn more robust temporal representations.
    """
    def __init__(self, input_size, hidden_size, n_channels=2, sfreq=100,
                 lstm_hidden_size=64, lstm_layers=2, n_attention_heads=4,
                 dropout=0.25, mask_ratio=0.5, block_size=None):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.block_size = block_size  # None = random, int = contiguous blocks

        # ---------- Encoder (processes only visible parts) ----------
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, hidden_size // 4, (n_channels, 1)),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),

            nn.Conv2d(hidden_size // 4, hidden_size // 2, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(hidden_size // 2, hidden_size, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

        self.temporal_reduction = input_size // 4

        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.embedding_layer = nn.Linear(lstm_hidden_size * 2, hidden_size)

        # ---------- Decoder (reconstructs masked parts) ----------
        # Special token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size * 2,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.reconstruction_layer = nn.Linear(lstm_hidden_size * 2, hidden_size)

        self.decoder_deconv = nn.Sequential(
            nn.Upsample(size=(1, 625), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),

            nn.Upsample(size=(1, 1250), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size // 2, hidden_size // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),

            nn.Conv2d(hidden_size // 4, n_channels, kernel_size=(1, 1))
        )

    def random_masking(self, x, mask_ratio, block_size=None):
        """
        Randomly masks parts of the temporal signal.

        Args:
            x: (B, C, T) input tensor
            mask_ratio: fraction of the signal to mask
            block_size: size of contiguous blocks to mask.
                       If None, point-wise (random) masking
                       If int, masks contiguous blocks of that size

        Returns:
            x_masked: signal with masked parts (set to 0)
            mask: binary mask (B, T) where 1=masked, 0=visible
        """
        B, C, T = x.shape

        if block_size is None or block_size == 1:
            # Point-wise (random) masking
            len_keep = int(T * (1 - mask_ratio))

            # Generate random indices for each sample in the batch
            noise = torch.rand(B, T, device=x.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # Create mask: 0 is visible, 1 is masked
            mask = torch.ones(B, T, device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)

        else:
            # Block-wise contiguous masking
            mask = torch.zeros(B, T, device=x.device)

            for b in range(B):
                # Calculate how many points to mask
                num_masked = int(T * mask_ratio)
                num_blocks = max(1, num_masked // block_size)

                # Generate random start positions for the blocks
                start_positions = torch.randint(0, max(1, T - block_size + 1),
                                               (num_blocks,), device=x.device)

                # Mask blocks
                for start in start_positions:
                    end = min(start + block_size, T)
                    mask[b, start:end] = 1

                # Adjust exact ratio if needed
                current_ratio = mask[b].sum() / T
                if current_ratio < mask_ratio:
                    # Add more random points
                    remaining = int((mask_ratio - current_ratio) * T)
                    unmasked_indices = torch.where(mask[b] == 0)[0]
                    if len(unmasked_indices) > 0:
                        to_mask = unmasked_indices[torch.randperm(len(unmasked_indices))[:remaining]]
                        mask[b, to_mask] = 1

        # Apply mask to the signal
        x_masked = x.clone()
        mask_expanded = mask.unsqueeze(1).expand_as(x)  # (B, C, T)
        x_masked = x_masked * (1 - mask_expanded)

        return x_masked, mask

    def encode(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.encoder_cnn(x)  # (B, H, 1, T')
        x = x.squeeze(2)         # (B, H, T')
        x = x.permute(0, 2, 1).contiguous()

        lstm_out, _ = self.encoder_lstm(x)  # (B, T', 2*LSTM_H)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        embedding = self.embedding_layer(attn_out)  # (B, T', H)
        return embedding

    def decode(self, embedding):
        x, _ = self.decoder_lstm(embedding)  # (B, T', 2*LSTM_H)
        x = self.reconstruction_layer(x)     # (B, T', H)

        x = x.permute(0, 2, 1).contiguous().unsqueeze(2)
        x = self.decoder_deconv(x)           # (B, C, ?, T)
        x = x.squeeze(2)                     # (B, C, T)
        return x

    def get_embedding(self, x):
        """Generates embedding without masking (for downstream tasks)"""
        embedding_seq = self.encode(x)          # (B, T', hidden_dim)
        embedding = embedding_seq.mean(dim=1)   # (B, hidden_dim)
        return embedding

    def forward(self, x, mask_ratio=None, block_size=None):
        """
        Forward pass with masking.

        Args:
            x: (B, C, T) input signal
            mask_ratio: masking ratio (uses self.mask_ratio if None)
            block_size: block size (uses self.block_size if None)

        Returns:
            reconstruction: (B, C, T) reconstructed signal (complete)
            embedding: (B, T', H) latent embeddings
            x_masked: (B, C, T) masked input
            mask: (B, T) binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        if block_size is None:
            block_size = self.block_size

        # 1. Mask input
        x_masked, mask = self.random_masking(x, mask_ratio, block_size)

        # 2. Encode (processes only visible parts)
        embedding = self.encode(x_masked)

        # 3. Decode (reconstruct complete signal)
        reconstruction = self.decode(embedding)

        return reconstruction, embedding, x_masked, mask


class EnhancedAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_channels=2, sfreq=100, lstm_hidden_size=64,
                 lstm_layers=2, n_attention_heads=4, dropout=0.25):
        super().__init__()

        self.n_channels = n_channels
        self.hidden_size = hidden_size

        self.cnn_block = nn.Sequential(
            # Initial spatial convolution
            nn.Conv2d(1, hidden_size // 4, (n_channels, 1)),
            nn.BatchNorm2d(hidden_size // 4),
            nn.ReLU(),

            # First temporal convolution
            nn.Conv2d(hidden_size // 4, hidden_size // 2, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            # Second temporal convolution
            nn.Conv2d(hidden_size // 2, hidden_size, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )

        # Calculate the output size after CNN block
        # After two MaxPool2d operations, the temporal dimension is reduced by factor of 4
        self.cnn_temporal_size = input_size // 4

        # LSTM block - input size is now 128 (from CNN output channels)
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # Modified to match CNN output channels
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Multi-head attention block
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,  # *2 because of bidirectional LSTM
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Projection head remains the same
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def get_embedding(self, x):
        """
        Forward pass through the network up to the embedding.
        Args:
            x (torch.Tensor): Input data of shape (batch, channels, time)
        """
        # Add channel dimension for 2D convolution: (batch, 1, channels, time)
        x = x.unsqueeze(1)

        # Pass through CNN block
        x = self.cnn_block(x.float())

        # Reshape for LSTM: (batch, time, channels)
        # Remove the redundant spatial dimension and permute
        x = x.squeeze(2)  # Remove the spatial dimension that was reduced to 1
        x = x.permute(0, 2, 1).contiguous()

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Multi-head attention
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling across time dimension
        embedding = attention_out.mean(dim=1)

        return embedding

    def forward(self, x):
        """
        Forward pass with projection head.
        Args:
            x (torch.Tensor): Input data of shape (batch, channels, time)
        Returns:
            torch.Tensor: Projected embeddings of shape (batch, hidden_size)
        """
        embedding = self.get_embedding(x)
        return self.projection(embedding)


class VariationalAttentionLSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_channels=2, sfreq=100,
                 lstm_hidden_size=64, lstm_layers=2, n_attention_heads=4,
                 dropout=0.25, latent_dim=None, prior=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim if latent_dim is not None else hidden_size

        self.backbone = AttentionLSTMAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            n_channels=n_channels,
            sfreq=sfreq,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            n_attention_heads=n_attention_heads,
            dropout=dropout,
        )

        self.fc_mu = nn.Linear(hidden_size, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, self.latent_dim)
        self.fc_expand = nn.Linear(self.latent_dim, hidden_size)

        self.prior = prior if prior is not None else StandardNormalPrior()

    def _pool_encode(self, x):
        h_seq = self.backbone.encode(x)
        h = h_seq.mean(dim=1)
        return h, h_seq.size(1)

    def encode_params(self, x):
        h, t_prime = self._pool_encode(x)
        # Clamp keeps exp(logvar) finite; the range is far wider than any
        # useful posterior variance, so it only guards against blow-ups.
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return self.fc_mu(h), logvar, t_prime

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar, t_prime = self.encode_params(x)
        z = self.reparameterize(mu, logvar)
        z_seq = self.fc_expand(z).unsqueeze(1).repeat(1, t_prime, 1)
        reconstruction = self.backbone.decode(z_seq)
        return reconstruction, mu, logvar, z

    def get_embedding(self, x):
        h, _ = self._pool_encode(x)
        return self.fc_mu(h)

    def kl(self, mu, logvar, free_bits=0.0):
        return self.prior.kl_divergence(mu, logvar, free_bits=free_bits)
