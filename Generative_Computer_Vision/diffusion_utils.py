from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the UNet architecture for diffusion with attention for global dependencies
class DiffusionUNet(nn.Module):
    def __init__(self, base_dim=32, n_channels=1, n_updown_blocks=4, n_middle_blocks=2):
        """base_dim is the number of channels after the first convolution, n_channels is the number of channels in the input image"""
        super(DiffusionUNet, self).__init__()
        
        self.n_channels = n_channels

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        self.conv_in = nn.Sequential(
            nn.Conv2d(n_channels+1, base_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.ReLU()
        )

        for i in range(n_updown_blocks):
            # Encoder layers
            in_channels = base_dim * (2**i)
            out_channels = base_dim * (2**(i+1))
            
            self.down_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.ReLU()
            ))
            
            # Decoder layers
            dec_in_channels = out_channels * 2  # We double the number of channels for skip connections
            dec_out_channels = in_channels
            
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(dec_in_channels, dec_out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, dec_out_channels),
                nn.ReLU(),
                nn.Conv2d(dec_out_channels, dec_out_channels, kernel_size=1, stride=1, padding=0), # Add a 1x1 conv to make future adaptations easier
                nn.GroupNorm(8, dec_out_channels),
                nn.ReLU()
            ))


        mid_block_width = base_dim * 2**(n_updown_blocks)

        self.middle = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_block_width, mid_block_width, 3, padding=1),
                nn.GroupNorm(8, mid_block_width),
                nn.ReLU()
            ) 
        for _ in range(n_middle_blocks)])

        self.conv_out = nn.Conv2d(2*base_dim, n_channels, kernel_size=3, padding=1)
    
    
    def forward(self, x, t, c=None):
        # Concatenate timestep as another channel in the image
        t = t.expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t], dim=1)

        x = self.conv_in(x)

        x_in = x

        # Store the hidden states as we downsample
        hidden_states = [x]
        for layer in self.down_blocks:
            x = layer(x)
            hidden_states.append(x)

        # Use residual connections in the middle
        for layer in self.middle:
            x = x + layer(x)

        # Use skip connections with the corresponding stored hidden states
        for layer, hidden_state in zip(reversed(self.up_blocks), reversed(hidden_states)):
            x = torch.cat([x, hidden_state], dim=1)
            x = layer(x)

        # Concatenate the original image with the stored hidden states
        x = torch.cat([x, x_in], dim=1)
        x = self.conv_out(x)
        
        return x


class ConditionalDiffusionUNet(DiffusionUNet):
    def __init__(self, base_dim=32, n_channels=1, n_updown_blocks=4, n_middle_blocks=2, n_classes=None, class_embedding_dim=16):
        # We will use concatenate the class embedding as some extra channels in the image

        super(ConditionalDiffusionUNet, self).__init__(
            base_dim=base_dim, n_channels=n_channels+class_embedding_dim, n_updown_blocks=n_updown_blocks, n_middle_blocks=n_middle_blocks
        )
        self.class_embedding = nn.Embedding(n_classes, class_embedding_dim)

        # we need to reset this as it is incorrectly set by the superclass constructor
        self.n_channels_img = n_channels
        self.n_classes = n_classes

    def forward(self, x, t, c):

        c = self.class_embedding(c)
        # Copy the embedding across the width and height of the image
        c = c.unsqueeze(-1).unsqueeze(-1)
        c = c.expand(-1, c.shape[1], x.shape[2], x.shape[3])

        x = torch.cat([x, c], dim=1)

        x = super().forward(x, t)

        x = x[:, :self.n_channels_img, :, :]

        return x
  
def validate_rectified_flow_conditional(model, val_dataloader):
    model.eval()
    model = model.to(device)
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_x, val_c = val_batch
            val_x = val_x.to(device)
            val_c = val_c.to(device)
            batch_size, _, _, _ = val_x.shape
            
            # Sample random timesteps
            val_t = torch.randint(0, 1000, (batch_size, 1, 1, 1)).to(device)
            
            # Add noise according to rectified flow
            val_noise = torch.randn_like(val_x)
            val_noisy_x = (1 - val_t / 1000) * val_x + (val_t / 1000) * val_noise
            
            # Predict direction
            val_pred_direction = model(val_noisy_x, val_t/1000, val_c)
            val_true_direction = val_noise - val_x
            
            val_loss += F.mse_loss(val_pred_direction, val_true_direction).item()
            
    val_loss /= len(val_dataloader)
    model.train()
    return val_loss
