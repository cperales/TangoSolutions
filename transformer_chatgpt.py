import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, img_size=6, embed_dim=64, nhead=8, num_layers=2, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.num_patches = img_size * img_size  # here each pixel is treated as a patch
        self.embed_dim = embed_dim
        
        # Use a convolutional layer to embed the image from in_channels to embed_dim.
        # Kernel size of 1 means no spatial mixing – only change in feature dimension.
        self.embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        
        # Create a learnable positional embedding (one for each patch in the flattened image).
        # Shape: (1, embed_dim, num_patches) so that it can be broadcasted.
        self.pos_embedding = nn.Parameter(torch.randn(1, embed_dim, self.num_patches))
        
        # Define a transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # This fully connected layer maps the transformer output for each patch
        # to a single value (logit for binary classification).
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape (6, 6, 3). If a batch is provided, then (B, 6, 6, 3).
        Returns:
            A tensor of shape (6, 6) with binary values {0, 1}.
        """
        # # Check if input has a batch dimension; if not, add one.
        # if x.ndim == 3:  # shape (6, 6, 3)
        #     # Rearrange to (batch, channels, height, width)
        #     x = x.permute(2, 0, 1).unsqueeze(0)
        # else:
        #     # Assume x shape is (B, 6, 6, 3) and rearrange to (B, 3, 6, 6)
        #     x = x.permute(0, 3, 1, 2)
        
        B = x.size(0)
        # Embed the image: output shape (B, embed_dim, 6, 6)
        x = self.embedding(x)
        
        # Flatten the spatial dimensions: output shape becomes (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Prepare data for the transformer: expected shape is (seq_len, B, embed_dim)
        x = x.permute(2, 0, 1)  # Now shape is (num_patches, B, embed_dim)
        
        # Process with transformer encoder
        x = self.transformer_encoder(x)
        
        # Revert the transformer output back to shape (B, num_patches, embed_dim)
        x = x.permute(1, 0, 2)
        
        # Apply the fully connected layer to each patch token
        logits = self.fc(x)  # shape: (B, num_patches, 1)
        logits = logits.squeeze(-1)  # shape: (B, num_patches)
        
        # Reshape back to image grid: (B, 6, 6)
        logits = logits.view(B, self.img_size, self.img_size)
        
        # Apply sigmoid to get probabilities and convert to binary output using threshold 0.5.
        probs = torch.sigmoid(logits)
        # binary_output = (probs > 0.5).int()
        
        # For a single image, return shape (6, 6)
        if B == 1:
            return probs.squeeze(0)
        return probs

# Example usage
if __name__ == '__main__':
    # Create an instance of the model.
    model = VisionTransformer()
    
    # Generate a dummy input image with shape (6, 6, 3)
    dummy_input = torch.randn(6, 6, 3)
    
    # Get the binary output
    output = model(dummy_input)
    
    print("Binary Output (6x6):")
    print(output)
