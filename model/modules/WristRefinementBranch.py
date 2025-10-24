import torch
import torch.nn as nn
import torch.nn.functional as F
import model.modules.StackedHourGlass as M


class WristRefinementBranch(nn.Module):
    def __init__(self, in_channels, out_channels=1, crop_size=32, beta=100, attention=True, use_roi=True):
        """
        Wrist refinement branch with two modes:
          1) ROI cropping around elbow for localized refinement
          2) Full feature map with attention (no cropping)
        
        Args:
            in_channels: channels from hourglass feature maps
            out_channels: number of wrist joints (default 1 per branch)
            crop_size: size of cropped ROI (square, e.g. 32x32)
            beta: temperature for soft-argmax (higher = sharper)
            attention: if True, apply attention over featurews/ROI
            use_roi: if True, crop ROI around elbow before refinement
        """
        super(WristRefinementBranch, self).__init__()
        self.crop_size = crop_size
        self.beta = beta
        self.use_attention = attention
        self.use_roi = use_roi
        self.in_channels = in_channels
        
        # CNN for ROI refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)  # wrist heatmap
        )

        # Global refinement (if use_roi=False)
        self.res1 = nn.Conv2d(in_channels + 2, 64, kernel_size=3, padding=1)
        self.res2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Optional attention
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=in_channels,
                                                   num_heads=4,
                                                   batch_first=True)

        
    def soft_argmax_2d(self, heatmap):
        """
        Differentiable soft-argmax for 2D heatmaps.
        Args:
            heatmap: (B,1,H,W)
        Returns:
            coords: (B,2) pixel coordinates in image space [0, H-1] x [0, W-1]

        """
        B, C, H, W = heatmap.shape
        assert C == 1, "Soft-argmax expects single-channel heatmap"

        heatmap_flat = heatmap.view(B, -1)
        prob = F.softmax(self.beta * heatmap_flat, dim=1)

        ys = torch.linspace(0, H-1, H, device=heatmap.device, dtype=heatmap.dtype)
        xs = torch.linspace(0, W-1, W, device=heatmap.device, dtype=heatmap.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(heatmap.dtype)  # (H*W,2)
        #print("prob shape:", prob.shape)         # Should be (B, H*W)
        #print("coords shape:", coords.shape)     # Should be (H*W, 2)
        #print("coords dtype:", coords.dtype)     # Should match heatmap.dtype
        exp_coords = torch.matmul(prob, coords)  # (B,2)
        #print("exp_coords shape:", exp_coords.shape)  # Should be (B, 2)
        return exp_coords

    def crop_roi(self, features, elbow_heatmap):
        """
        Crop a differentiable ROI around elbow using soft-argmax.
        Args:
            features: (B,C,H,W) feature map
            elbow_heatmap: (B,1,H,W)
        Returns:
            roi: (B,C,crop_size,crop_size)
        """
        B, C, H, W = features.shape
        #print("elbow_heatmap shape:", elbow_heatmap.shape)

        # Get elbow coordinates in [-1,1]
        
        elbow_coords = self.soft_argmax_2d(elbow_heatmap)  # (B,2)
        
        #elbow_coords = elbow_coords.view(B, 1, 1, 2)
        x = elbow_coords[:, 0]  # shape: (B,)
        y = elbow_coords[:, 1]  # shape: (B,)

        x_norm = (x / (W - 1)) * 2 - 1
        y_norm = (y / (H - 1)) * 2 - 1
        elbow_coords_norm = torch.stack([x_norm, y_norm], dim=1).view(B, 1, 1, 2)



        # Base grid for crop
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.crop_size, device=features.device),
            torch.linspace(-1, 1, self.crop_size, device=features.device),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B,1,1,1)

        # Shift grid to elbow center
        grid = base_grid + elbow_coords_norm
        grid = torch.clamp(grid, -1, 1)

        roi = F.grid_sample(features, grid, align_corners=True)
        return roi
    

        
    def forward(self, features, elbow_heatmap, forearm_map, out_size=None):
        """
        Forward pass for wrist refinement.
        Args:
            features: (B,C,H,W)
            elbow_heatmap: (B,1,H,W)
            forearm_map: (B,1,H,W)
            out_size: (H,W) to upsample wrist heatmap to match main heatmaps
        Returns:
            wrist_heatmap: (B,out_channels,H_out,W_out)
        """
        if self.use_roi:
            # --- ROI cropping mode ---
            roi = self.crop_roi(features, elbow_heatmap)

            # Optional attention
            if self.use_attention:
                B, C, H, W = roi.shape
                roi_flat = roi.view(B, C, -1).transpose(1, 2)  # (B,HW,C)
                roi_attn, _ = self.attention(roi_flat, roi_flat, roi_flat)
                roi = roi_attn.transpose(1, 2).view(B, C, H, W)

            wrist_heatmap = self.refine(roi)


            # Upsample to match main heatmaps
            if out_size is not None:
                wrist_heatmap = F.interpolate(wrist_heatmap, size=out_size, mode='bilinear', align_corners=True)

            
            return wrist_heatmap, forearm_map, roi
        
        else:
            x = torch.cat([features, elbow_heatmap, forearm_map], dim=1)
            x = F.relu(self.res1(x))
            x = F.relu(self.res2(x))
            x = F.relu(self.res3(x))

            if self.use_attention:
                B, C, H, W = x.shape
                x_flat = x.view(B, C, -1).transpose(1, 2)
                attn_out, _ = self.attention(x_flat, x_flat, x_flat)
                attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
                x = x * attn_out

            out = self.out_conv(x)
            return out


def heatmap_to_coord(heatmap):
    """
    Argmax to convert 1-channel heatmap to (x,y) coordinates
    heatmap: (B,1,H,W)
    returns: (B,2) tensor
    """
    B, _, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, -1)
    idx = heatmap_flat.argmax(dim=1)
    y = idx // W
    x = idx % W
    return torch.stack([x.float(), y.float()], dim=1)


def generate_forearm_mask(elbow_coords, wrist_coords, H, W, sigma=4.0):
    """
    Generate Gaussian forearm mask along elbow->wrist line
    """
    # elbow_coords/wrist_coords: (B,2) in pixel coords (x,y)
    B = elbow_coords.shape[0]
    device = elbow_coords.device
    dtype = elbow_coords.dtype

    xs = torch.arange(0, W, device=device, dtype=dtype).view(1,1,W).expand(B,H,W)
    ys = torch.arange(0, H, device=device, dtype=dtype).view(1,H,1).expand(B,H,W)

    x_e = elbow_coords[:,0].view(B,1,1)
    y_e = elbow_coords[:,1].view(B,1,1)
    x_w = wrist_coords[:,0].view(B,1,1)
    y_w = wrist_coords[:,1].view(B,1,1)

    dx = x_w - x_e
    dy = y_w - y_e
    length_sq = dx*dx + dy*dy + 1e-6

    t = ((xs - x_e)*dx + (ys - y_e)*dy) / length_sq
    t = t.clamp(0,1)
    proj_x = x_e + t*dx
    proj_y = y_e + t*dy

    dist_sq = (xs - proj_x)**2 + (ys - proj_y)**2
    mask = torch.exp(-dist_sq/(2*sigma**2)).unsqueeze(1)  # (B,1,H,W)
    return mask