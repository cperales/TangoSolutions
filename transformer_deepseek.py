import torch
import torch.nn as nn

class ImageTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        # Project each pixel's 3 channels into d_model dimensions
        self.proj = nn.Linear(3, d_model)
        # Learnable positional embeddings for 6x6 grid
        self.pos_embed = nn.Parameter(torch.randn(1, 6*6, d_model))
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Decoder to produce logits for each pixel
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        # Input shape: (batch_size, 3, 6, 6)
        batch_size = x.size(0)
        # Reshape and project to embeddings
        x = x.permute(0, 2, 3, 1).reshape(batch_size, 6*6, 3)  # (batch, 36, 3)
        x = self.proj(x)  # (batch, 36, d_model)
        # Add positional embeddings
        x += self.pos_embed
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, 36, d_model)
        # Decode to logits
        x = self.decoder(x).squeeze(-1)  # (batch, 36)
        # Reshape to (batch, 6, 6)
        x = x.view(batch_size, 6, 6)
        return torch.sigmoid(x)


# Example usage
if __name__ == "__main__":
    # Create a random input image (batch_size=1, 3 channels, 6x6)
    input_image = torch.randn(1, 3, 6, 6)
    model = ImageTransformer()
    output = model(input_image)
    print("Output shape:", output.shape)  # Should be (1, 6, 6)

    # To get binary output (0 or 1), apply sigmoid and threshold
    binary_output = (torch.sigmoid(output) > 0.5).int()
    print("Binary output shape:", binary_output.shape)