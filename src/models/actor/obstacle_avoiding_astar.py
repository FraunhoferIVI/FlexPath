import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SemanticSegmenterOutput


class ObstacleAvoidingAstar(nn.Module):
    def __init__(self, clearance_distance, h_w: float = 10.0):
        super().__init__()

        self.clearance_distance = int(clearance_distance)
        self.h_w = h_w

        dim = 2 * (self.clearance_distance - 1) + 1
        self.conv_padding = dim // 2

        ones_kernel = torch.ones(size=(1, 1, dim, dim))

        self.register_buffer("ones_kernel", ones_kernel)

        semob_kernel = torch.ones(size=(1, 1, 7, 7))

        self.register_buffer("semobkernel", semob_kernel)

    def octile_distance(self, grid_shape, goals):

        """

        grid_shape: (H, W)
        goals: (B, 2) tensor of goal coordinates (y, x) per batch
        returns: (B, H, W) tensor of octile distances to goal

        """

        B = goals.shape[0]
        H, W = grid_shape

        device = goals.device

        # Create grid coordinates
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device),
                                        indexing='ij')  # (H, W)
        grid_coords = torch.stack([grid_y, grid_x], dim=0)  # (2, H, W)
        grid_coords = grid_coords.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 2, H, W)

        # Goal coordinates per batch
        goal_coords = goals[:, :, None, None]  # (B, 2, 1, 1)

        # Compute dx, dy
        delta = (grid_coords - goal_coords).abs()  # (B, 2, H, W)
        dx = delta[:, 1]  # (B, H, W)
        dy = delta[:, 0]  # (B, H, W)

        # Octile distance formula
        sqrt2_minus_1 = torch.sqrt(torch.tensor(2.0, device=device)) - 1.0
        octile_dist = dx + dy + (sqrt2_minus_1 - 1) * torch.min(dx, dy)

        return octile_dist
    
    def extract_coordinates(self, one_hot_maps):
        # one_hot_maps: [B,1,H,W]
        B, _, H, W = one_hot_maps.shape

        # Flatten H,W dimensions
        flat_maps = one_hot_maps.view(B, -1)  # [B, H*W]

        # Get linear indices of the '1's
        indices = flat_maps.argmax(dim=1)  # [B]

        # Convert linear index to 2D coordinates
        y = indices // W
        x = indices % W

        coords = torch.stack([y, x], dim=1)  # [B, 2] as (y,x)
        return coords

    def forward(self, images: torch.FloatTensor):
        """
        images: (B,3,H,W)
        returns: (B,H,W) heuristic grid combining octile distance and obstacle clearance
        """

        device = images.device

        # Identify obstacle pixels
        obstacle_color = torch.tensor([76/255, 76/255, 255/255], device=device)
        obstacle_map = torch.sigmoid((images - obstacle_color.view(3,1,1)).abs()).sum(1, keepdim=True)
        obstacle_map = (obstacle_map < 1.51).float()  # 1 = obstacle

        semob_color = torch.tensor([100/255, 100/255, 255/255], device=device)
        semob_map = torch.sigmoid((images - semob_color.view(3,1,1)).abs()).sum(1, keepdim=True)
        semob_map = (semob_map < 1.51).float()

        end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=images.device)  # BLUE
        end_map = torch.sigmoid((images - end_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        end_map = (end_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies
        ends = self.extract_coordinates(end_map)

        B, _, H, W = obstacle_map.shape
        # dist_map = torch.zeros((B, H, W), device=device)

        # # Find obstacle coordinates per batch
        # for b in range(B):
        #     obstacle_coords = torch.nonzero(obstacle_map[b,0], as_tuple=False).float()  # (N_obst, 2)
        #     if obstacle_coords.shape[0] == 0:
        #         dist_map[b] = torch.full((H,W), float('inf'), device=device)
        #         continue

        #     # Create grid coordinates
        #     grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        #     grid_coords = torch.stack([grid_y, grid_x], dim=-1).view(-1,2).float()  # (H*W,2)

        #     # Compute all pairwise distances to obstacles
        #     dists = torch.cdist(grid_coords.unsqueeze(0), obstacle_coords.unsqueeze(0)).squeeze(0)  # (H*W, N_obst)

        #     # Minimum distance for each pixel
        #     min_dists, _ = torch.min(dists, dim=1)
        #     dist_map[b] = min_dists.view(H,W)

        expanded_obstacle_map_normalobstacles = F.conv2d(
            obstacle_map,
            weight=self.ones_kernel,
            stride=1,
            padding=self.conv_padding
        )  # expand obstacles by given distance

        expanded_obstacle_map_semob = F.conv2d(
            semob_map,
            weight=self.semobkernel,
            stride=1,
            padding=3
        )  # expand obstacles by given distance

        expanded_obstacle_map = expanded_obstacle_map_normalobstacles + expanded_obstacle_map_semob

        expanded_obstacle_map = expanded_obstacle_map.clamp_max(max=1.0)  # reduce to whether obstacle in neighborhood or not

        octile = self.octile_distance(grid_shape=(H, W), goals=ends)
        obstacle_h = (self.h_w * expanded_obstacle_map).squeeze(1)

        heuristic = octile + obstacle_h

        return SemanticSegmenterOutput(
            loss=None,
            logits=heuristic,
            hidden_states=None,
            attentions=None
        )