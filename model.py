import math
import torch
import torch.nn as nn


# ==================== SINUSOIDAL TIME EMBEDDING ====================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model=128, max_t=500):
        super().__init__()
        self.d_model = d_model
        self.max_t = max_t

    def forward(self, t):
        """
        Args:
            t: (B,) or (B,1) Tensor containing time steps
        Returns:
            (B, d_model) Sinusoidal embeddings
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # Ensure shape is (B,1)

        device = t.device

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(self.max_t) / self.d_model)
        )

        sin_emb = torch.sin(t * div_term)  # (B, d_model/2)
        cos_emb = torch.cos(t * div_term)  # (B, d_model/2)

        return torch.cat([sin_emb, cos_emb], dim=-1)  # (B, d_model)


class StyleEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.style_heads = nn.ModuleList(
            [
                nn.Linear(128, out_c * out_l)
                for out_c, out_l in [(1, 64), (1, 32), (1, 16)]
            ]
        )

    def forward(self, style_bands):
        return [
            self.style_heads[i](style_bands[i]).view(style_bands[i].shape[0], *size)
            for i, size in enumerate([(1, 64), (1, 32), (1, 16)])
        ]


class MultiResUNet1D(nn.Module):
    def __init__(self, d_model=128, num_style=3, T=500):
        super().__init__()
        self.num_style = num_style
        self.time_embed = SinusoidalTimeEmbedding(d_model, max_t=T)
        self.style_embed = StyleEmbedding()

        # Define downsampling layers (encoder)
        self.down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, kernel_size=3, stride=2, padding=1
                    ),
                    nn.GELU(),
                )
                for in_channels, out_channels in [(6, 4), (4, 6), (6, 8)]
            ]
        )

        # Bottleneck - No explicit t_emb injection here, already merged with content
        self.bottleneck = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=3, padding=1), nn.GELU()
        )

        # Define upsampling layers (decoder) using bilinear upsampling
        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=True),
                    nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GELU(),  # Keep only one non-linearity after feature refinement
                )
                for in_channels, out_channels in [
                    (4 + 4 + 1, 6),
                    (6 + 4 + 1, 4),
                    (8 + 8 + 1, 4),
                ]
            ]
        )

        # Fully connected final layer instead of Conv1d
        self.final_layer = nn.Sequential(
            nn.Linear(6 * 128, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 128),
        )

    def forward(self, z, t, content, style_bands):
        # z: noisy time series (B, 1, d), t: (B), content: (B, 1, d), style bands: [(B, 1, d)]
        B, _, L = z.shape
        # Compute time embedding and reshape at the start
        t_emb = self.time_embed(t).view(B, 1, 128)  # Convert to (B, 1, 128) to match z
        z = torch.cat(
            [z, t_emb, content, style_bands[0], style_bands[1], style_bands[2]], dim=1
        )
        # content_emb = content.view(B, 8, 16)  # Reshape to match bottleneck
        style_bands = self.style_embed(style_bands)
        # Encoder (downsampling)
        encoder_outputs = []
        x = z  # Start with noisy input
        for down in self.down_blocks:
            x = down(x)  # Downsample
            encoder_outputs.append(x)  # Store for skip connections
            # print(x.shape)
        # Bottleneck (Content already merged with t_emb earlier)
        x = self.bottleneck(x)
        # x = torch.cat([x, content_emb], dim=1)
        # Decoder (upsampling with style refinement)
        for i in range(self.num_style - 1, -1, -1):
            x = torch.cat(
                [x, encoder_outputs[i], style_bands[i]], dim=1
            )  # Skip connection
            x = self.up_blocks[i](x)  # Upsample
            # print(x.shape)
        x = x + z
        # Flatten before fully connected layer
        x = x.view(B, -1)
        # Final prediction using fully connected layers
        noise_pred = self.final_layer(x).view(B, 1, 128)
        return noise_pred
