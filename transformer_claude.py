import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=2, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, h/patch_size, w/patch_size)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention block
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attention(x, x, x)
        x = residual + x_attn
        
        # Feed-forward block
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class ImageTransformer(nn.Module):
    def __init__(self, img_size=6, patch_size=2, in_channels=3, 
                 embed_dim=64, num_heads=4, ff_dim=128, 
                 num_layers=4, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer layers
        self.transformers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Binary image reconstruction head
        # We need to go from embedded space back to original image dimensions
        self.img_size = img_size
        self.patch_size = patch_size
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size),
            nn.Sigmoid()
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Apply transformer layers
        for transformer in self.transformers:
            x = transformer(x)
        
        # Process output to create binary image
        patches_dim = self.img_size // self.patch_size
        reconstructed = []
        
        for i in range(x.shape[1]):
            patch_flat = self.output_head(x[:, i])  # (batch_size, patch_size*patch_size)
            patch = patch_flat.reshape(batch_size, self.patch_size, self.patch_size)
            reconstructed.append(patch)
        
        # Reshape patches back to image form
        reconstructed_patches = torch.stack(reconstructed, dim=1)  # (batch_size, n_patches, patch_size, patch_size)
        reconstructed_patches = reconstructed_patches.reshape(
            batch_size, patches_dim, patches_dim, self.patch_size, self.patch_size
        )
        
        # Rearrange to get the full image
        reconstructed_img = torch.zeros(batch_size, self.img_size, self.img_size, device=x.device)
        for i in range(patches_dim):
            for j in range(patches_dim):
                reconstructed_img[:, i*self.patch_size:(i+1)*self.patch_size, 
                                    j*self.patch_size:(j+1)*self.patch_size] = reconstructed_patches[:, i, j]
        
        # # Apply threshold to get binary output
        # binary_output = (reconstructed_img > 0.5).float()
        
        return reconstructed_img

# Example usage
if __name__ == "__main__":
    # Create random test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 6, 6)
    
    # Create model
    model = ImageTransformer(img_size=6, patch_size=2, in_channels=3)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output example:\n{output[0]}")
    print(f"Contains only 0s and 1s: {torch.all((output == 0) | (output == 1))}")