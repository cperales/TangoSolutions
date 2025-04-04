import torch
from torch import nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=36):
        """
        Positional encoding for transformer
        
        Args:
            d_model: Embedding size
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer makes the tensor part of the module state without being a parameter
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        return x + self.pe[:, :x.size(1)]

class TangoTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, dim_feedforward=512):
        super(TangoTransformer, self).__init__()
        
        # Initial convolutional embedding
        self.conv_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Feature dimension will be d_model
        self.d_model = d_model
        
        # Transform 2D board to sequence
        # For a 6x6 board, sequence length = 36
        self.seq_len = 36
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )
        
        # Transformer decoder part
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Learnable query for transformer decoder
        self.query_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initial convolution to get features
        # x: [batch_size, 3, 6, 6]
        features = self.conv_embed(x)  # [batch_size, d_model, 6, 6]
        
        # Reshape to sequence form
        # [batch_size, d_model, 6, 6] -> [batch_size, d_model, 36]
        seq_features = features.view(batch_size, self.d_model, -1)
        
        # Transpose to [batch_size, seq_len, d_model]
        seq_features = seq_features.transpose(1, 2)
        
        # Add positional encoding
        pos_encoded = self.positional_encoding(seq_features)
        
        # Transformer encoder
        # [batch_size, seq_len, d_model]
        memory = self.transformer_encoder(pos_encoded)
        
        # Repeat query embedding for batch size
        query_embed = self.query_embed.repeat(batch_size, 1, 1)
        
        # Transformer decoder
        # [batch_size, seq_len, d_model]
        decoded = self.transformer_decoder(query_embed, memory)
        
        # Project to get probabilities
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, 1]
        outputs = self.projection(decoded).squeeze(-1)  # [batch_size, seq_len]
        
        # Reshape back to board dimensions
        # [batch_size, 36] -> [batch_size, 6, 6]
        outputs = outputs.view(batch_size, 6, 6)
        
        return outputs


class EnsembleCNN(nn.Module):
    def __init__(self, n):
        super().__init__()  # Call the parent class constructor
        self.ensemble = nn.ModuleList([TangoCNN() for _ in range(n)])
        self.alpha = torch.ones(n) / n
    
    def forward(self, x):
        batch_size = x.shape[0]
        out = torch.zeros((batch_size, 6, 6))
        for i, base_learner in enumerate(self.ensemble):
            out += self.alpha[i] * base_learner(x)
        return out


class TangoCNN(nn.Module):
    def __init__(self):
        super(TangoCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.residual1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        
        # Final prediction layer
        self.final_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_layers(x)
        
        # Residual connections
        residual = x
        x = self.residual1(x)
        x = x + residual
        x = nn.functional.relu(x)
        
        residual = x
        x = self.residual2(x)
        x = x + residual
        x = nn.functional.relu(x)
        
        x = self.final_layer(x)
        x = self.sigmoid(x)
        
        # Return output as (batch_size, 6, 6)
        return x.view(-1, 6, 6)
