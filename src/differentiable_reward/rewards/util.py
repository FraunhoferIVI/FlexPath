import torch
import torch.nn.functional as F

import math


"""
Utility functions for differentiable reward computations.

This module provides:
- is_collision: detect overlap between predicted path and obstacles.
- is_connected: differentiably check connectivity between start and end on a predicted path.
- compute_path_deviation_penalties: soft distance-based penalty for deviating from a smoothed path.
- compute_soft_obstacle_distances: differentiable proximity penalties to obstacles.
- compute_path_cost_approximation: local path-length approximation via convolution.

All functions work on PyTorch tensors and preserve differentiability.

"""


DISTS_CACHE = {}


def is_collision(
    path_pred: torch.Tensor,
    obstacle_map: torch.Tensor
):

    """

    Parameters
    - path_pred: torch.Tensor, shape [B,1,H,W], values in [0,1]
        Predicted path heatmap per batch.
    - obstacle_map: torch.Tensor, shape [B,1,H,W] or broadcastable to it
        Binary or soft map indicating obstacles (1 = obstacle).

    Returns
    - penalty: torch.Tensor, shape [B,1]
        Per-batch maximum overlap between path_pred and obstacle_map. Higher means collision.

    Short description
    - Computes whether the predicted path overlaps obstacles by element-wise multiplication
      and then reducing over spatial dimensions.

    Detailed steps
    1. Compute overlap = path_pred * obstacle_map (element-wise) -> [B,1,H,W].
    2. Reduce across spatial dims (H,W) using torch.amax to get the maximum overlap per batch.
       This is a conservative indicator: if any pixel has high overlap, the batch is penalized.

    """

    # overlap shape: [B,1,H,W]
    overlap = path_pred * obstacle_map
    # penalty shape: [B,1] (reduced over H and W)
    penalty = torch.amax(overlap, dim=(2, 3))  # Reduce over spatial dimensions
    return penalty


def is_connected(
    path_pred: torch.Tensor,
    start: torch.Tensor, 
    end: torch.Tensor, 
    steps: int = 125, 
    sharpness: int = 5
):

    """

    Parameters
    - path_pred: torch.Tensor, shape [B,1,H,W], values in [0,1]
        Soft predicted path mask (confidence for each pixel).
    - start: torch.Tensor, shape [B,1,H,W]
        One-hot or soft mask indicating the start location(s) per batch.
    - end: torch.Tensor, shape [B,1,H,W]
        One-hot or soft mask indicating the end location(s) per batch.
    - steps: int
        Number of iterative dilation steps to attempt to reach the end.
    - sharpness: float
        Scaling applied before sigmoid to sharpen the final connectivity score.

    Returns
    - connectivity_score: torch.Tensor, shape [B]
        Value in (0,1) indicating how connected the start and end are along path_pred.
        Values near 1 indicate connected, values near 0 indicate disconnected.

    Short description
    - Performs a differentiable seeded-region growth (using max-pooling) from the start location,
      constrained to propagate only along the predicted path, and then evaluates whether the end
      location was reached.

    Detailed steps
    1. Extract batch size and spatial dims from path_pred.
    2. Convert start/end soft masks into representative coordinates by argmax over spatial dims.
       - start_idx/end_idx have shape [B], indices in flattened spatial domain.
       - Convert flattened indices to (x,y) via modulo/division with width.
    3. Create seeds tensor with zeros and set the seed location at the start coordinate.
       - seeds shape: [B,1,H,W]
    4. To ensure the end is considered reachable even when path doesn't place mass exactly on it,
       inject the end location into a copy of path_pred (path_pred_with_end_included).
    5. Iteratively dilate the reachable region using F.max_pool2d with kernel_size=3, stride=1, padding=1.
       After each dilation step, mask by path_pred_with_end_included to only propagate along predicted path.
    6. After 'steps' iterations, read the reachable value at the end coordinate.
    7. Return a sharpened sigmoid of (end_val - 0.5) to produce a score in (0,1). Sharpness controls
       how binary the output becomes.

    Notes about shapes while computing:
    - path_pred: [B,1,H,W]
    - start_flat/end_flat: [B, H*W]
    - start_idx/end_idx: [B]
    - start_x,start_y,end_x,end_y: [B] (integer coordinates per batch)
    - seeds/reachable: [B,1,H,W]
    - end_vals: [B] (float reachable value at end coordinates)

    """

    B, _, H, W = path_pred.shape

    # Extract start and end coordinates using argmax along spatial dimensions
    # For each batch, find the location of maximum value in the start and end masks
    start_flat = start.view(B, -1)  # [B, H*W]
    end_flat = end.view(B, -1)      # [B, H*W]

    start_idx = torch.argmax(start_flat, dim=1)  # [B] flattened indices
    end_idx = torch.argmax(end_flat, dim=1)      # [B]

    # Convert flattened indices back to 2D coordinates (x, y)
    start_x = start_idx % W  # [B] x-coordinate
    start_y = start_idx // W  # [B] y-coordinate
    end_x = end_idx % W      # [B]
    end_y = end_idx // W     # [B]

    # Initialize seeds with zeros
    seeds = torch.zeros(
        size=path_pred.shape,  # [B,1,H,W]
        device=path_pred.device,
        dtype=torch.float32
    )

    # Include the end location in the path copy so end can be "reached" even if the predicted
    # path stops adjacent to the end pixel.
    path_pred_with_end_included = torch.clone(path_pred)  # [B,1,H,W]
    path_pred_with_end_included[torch.arange(B), 0, end_y, end_x] = 1.0  # broadcast index by batch

    # Place a seed at the start point (use the batch indices and the coordinates)
    seeds[torch.arange(B), 0, start_y, start_x] = 1.0  # seeds now contain 1 at start locations

    reachable = seeds.clone()  # [B,1,H,W]

    for _ in range(steps):
        # Dilate reachable area using 3x3 max-pool (differentiable morphological dilation)
        reachable = F.max_pool2d(
            input=reachable, 
            kernel_size=3, 
            stride=1,
            padding=1
        )
        # Only keep propagation where path_pred_with_end_included allows (i.e., along the predicted path)
        reachable = reachable * path_pred_with_end_included  # [B,1,H,W]

    # Read connectivity at the end point (extract the value at the end coordinates)
    end_vals = reachable[torch.arange(B), 0, end_y, end_x]  # [B]
    # Sharpen and map to (0,1) with sigmoid; subtracting 0.5 centers threshold at 0.5
    return  F.sigmoid(sharpness * (end_vals - 0.5))

def is_connected__waypoint(
    path_pred: torch.Tensor,
    start: torch.Tensor, 
    end: torch.Tensor, 
    waypoint: torch.Tensor,
    steps: int = 125, 
    sharpness: int = 5
):

    """

    Parameters
    - path_pred: torch.Tensor, shape [B,1,H,W], values in [0,1]
        Soft predicted path mask (confidence for each pixel).
    - start: torch.Tensor, shape [B,1,H,W]
        One-hot or soft mask indicating the start location(s) per batch.
    - end: torch.Tensor, shape [B,1,H,W]
        One-hot or soft mask indicating the end location(s) per batch.
    - steps: int
        Number of iterative dilation steps to attempt to reach the end.
    - sharpness: float
        Scaling applied before sigmoid to sharpen the final connectivity score.

    Returns
    - connectivity_score: torch.Tensor, shape [B]
        Value in (0,1) indicating how connected the start and end are along path_pred.
        Values near 1 indicate connected, values near 0 indicate disconnected.

    Short description
    - Performs a differentiable seeded-region growth (using max-pooling) from the start location,
      constrained to propagate only along the predicted path, and then evaluates whether the end
      location was reached.

    Detailed steps
    1. Extract batch size and spatial dims from path_pred.
    2. Convert start/end soft masks into representative coordinates by argmax over spatial dims.
       - start_idx/end_idx have shape [B], indices in flattened spatial domain.
       - Convert flattened indices to (x,y) via modulo/division with width.
    3. Create seeds tensor with zeros and set the seed location at the start coordinate.
       - seeds shape: [B,1,H,W]
    4. To ensure the end is considered reachable even when path doesn't place mass exactly on it,
       inject the end location into a copy of path_pred (path_pred_with_end_included).
    5. Iteratively dilate the reachable region using F.max_pool2d with kernel_size=3, stride=1, padding=1.
       After each dilation step, mask by path_pred_with_end_included to only propagate along predicted path.
    6. After 'steps' iterations, read the reachable value at the end coordinate.
    7. Return a sharpened sigmoid of (end_val - 0.5) to produce a score in (0,1). Sharpness controls
       how binary the output becomes.

    Notes about shapes while computing:
    - path_pred: [B,1,H,W]
    - start_flat/end_flat: [B, H*W]
    - start_idx/end_idx: [B]
    - start_x,start_y,end_x,end_y: [B] (integer coordinates per batch)
    - seeds/reachable: [B,1,H,W]
    - end_vals: [B] (float reachable value at end coordinates)

    """

    B, _, H, W = path_pred.shape

    # Extract start and end coordinates using argmax along spatial dimensions
    # For each batch, find the location of maximum value in the start and end masks
    start_flat = start.view(B, -1)  # [B, H*W]
    end_flat = end.view(B, -1)      # [B, H*W]
    waypoint_flat = waypoint.view(B, -1)      # [B, H*W]

    start_idx = torch.argmax(start_flat, dim=1)  # [B] flattened indices
    end_idx = torch.argmax(end_flat, dim=1)      # [B]
    waypoint_idx = torch.argmax(waypoint_flat, dim=1)      # [B]

    # Convert flattened indices back to 2D coordinates (x, y)
    start_x = start_idx % W  # [B] x-coordinate
    start_y = start_idx // W  # [B] y-coordinate
    end_x = end_idx % W      # [B]
    end_y = end_idx // W     # [B]
    waypoint_x = waypoint_idx % W      # [B]
    waypoint_y = waypoint_idx // W     # [B]

    # Initialize seeds with zeros
    seeds = torch.zeros(
        size=path_pred.shape,  # [B,1,H,W]
        device=path_pred.device,
        dtype=torch.float32
    )

    # Include the end location in the path copy so end can be "reached" even if the predicted
    # path stops adjacent to the end pixel.
    path_pred_with_end_included = torch.clone(path_pred)  # [B,1,H,W]
    path_pred_with_end_included[torch.arange(B), 0, end_y, end_x] = 1.0  # broadcast index by batch

    # Place a seed at the start point (use the batch indices and the coordinates)
    seeds[torch.arange(B), 0, start_y, start_x] = 1.0  # seeds now contain 1 at start locations

    reachable = seeds.clone()  # [B,1,H,W]

    for _ in range(steps):
        # Dilate reachable area using 3x3 max-pool (differentiable morphological dilation)
        reachable = F.max_pool2d(
            input=reachable, 
            kernel_size=3, 
            stride=1,
            padding=1
        )
        # Only keep propagation where path_pred_with_end_included allows (i.e., along the predicted path)
        reachable = reachable * path_pred_with_end_included  # [B,1,H,W]

    # Read connectivity at the end point (extract the value at the end coordinates)
    end_vals = reachable[torch.arange(B), 0, end_y, end_x]  # [B]
    waypoint_vals = reachable[torch.arange(B), 0, waypoint_y, waypoint_x]  # [B]
    # Sharpen and map to (0,1) with sigmoid; subtracting 0.5 centers threshold at 0.5
    return  (F.sigmoid(sharpness * (end_vals - 0.5)) + F.sigmoid(sharpness * (waypoint_vals - 0.5))) / 2


def compute_soft_obstacle_distances(
    path_pred: torch.Tensor, 
    obstacle_grid: torch.Tensor, 
    desired_min_dist: float, 
    tau: float = 25.0
):

    """

    Parameters
    - path_pred: torch.Tensor, shape [B,1,H,W]
        Predicted path confidence map.
    - obstacle_grid: torch.Tensor, shape [B,1,H,W]
        Binary or soft map indicating obstacles.
    - desired_min_dist: float
        Distance threshold; distances below this incur penalties.
    - tau: float
        Temperature parameter controlling the sharpness of the soft-min.

    Returns
    - mean_penalty: torch.Tensor, shape [B,1]
        Negative mean proximity penalty per batch scaled by number of path pixels.
    - max_penalty: torch.Tensor, shape [B,1]
        Negative max proximity penalty per batch.

    Short description
    - Computes a differentiable estimate of the distance from each path pixel to the nearest
      obstacle using a soft-min over obstacle distances, then penalizes pixels closer than
      desired_min_dist.

    Detailed steps
    1. Build a normalized coordinate grid coords with values in [0,1] for numerical stability.
       coords shape: [H,W,2].
    2. Flatten coordinates for obstacles to compute pairwise distances between every path pixel
       and every obstacle pixel.
    3. Compute pairwise Euclidean distances dists of shape [B,1,H,W,HW].
    4. Mask distances with obstacle presence: masked_dists = dists * obstacle_w + (1 - obstacle_w) * large_value.
       This ensures non-obstacle locations don't affect the soft-min.
    5. Apply a softmin (via log-sum-exp) over the obstacle dimension to get a differentiable
       approximation of the minimum distance to any obstacle: d_closest_obstacle -> [B,1,H,W].
       The factor ((H+W)/2) rescales distances relative to grid size.
    6. Compute proximity = ReLU(desired_min_dist - d_closest_obstacle) and normalize by desired_min_dist.
    7. Weight proximity by path_pred to focus penalty only on path pixels.
    8. Compute mean and max penalties; mean is multiplied by pixel count to retain scale.

    """

    B, _, H, W = path_pred.shape
    device = path_pred.device

    # Sum over pixels in path_pred per batch: [B]
    pixel_sums = torch.sum(path_pred, dim=(1, 2, 3))

    # 1. Coordinate grid normalized to [0,1]; coords: [H,W,2]
    if (H, W) not in DISTS_CACHE.keys():
        yy = torch.linspace(0, 1, H, device=device)
        xx = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        coords = torch.stack([yy, xx], dim=-1)  # [H,W,2]
        
        
        # Flatten coords for obstacle points: [HW,2]
        coords_flat = coords.reshape(-1, 2)             # [HW,2]

        # Reshape coords for pairwise computations:
        # path_coords: [1,1,H,W,2]  ; obstacle_coords: [1,1,1,1,HW,2]
        path_coords = coords.reshape(1, 1, H, W, 2)      # [1,1,H,W,2]
        obstacle_coords = coords_flat.reshape(1, 1, 1, 1, -1, 2)  # [1,1,1,1,HW,2]

        # 2. Pairwise distances: subtract and norm -> [B,1,H,W,HW]
        DISTS_CACHE[(H, W)] = ((path_coords.unsqueeze(4) - obstacle_coords)**2).sum(-1).sqrt()
            
    dists = DISTS_CACHE[(H, W)]

    # obstacle_flat: [B,1,HW]
    obstacle_flat = obstacle_grid.reshape(B, 1, -1) # [B,1,HW]

    # 3. Obstacle weighting: [B,1,1,1,HW]
    obstacle_w = obstacle_flat.unsqueeze(2).unsqueeze(2)   # [B,1,1,1,HW]

    # Mask non-obstacle distances with a large value so they don't affect soft-min
    masked_dists = dists * obstacle_w + (1 - obstacle_w) * 1e6

    # 4. Softmin over obstacle dimension:
    # Multiply distances by (H+W)/2 to scale with grid size before applying soft-min (tau temp)
    d_closest_obstacle = -torch.logsumexp(-tau * masked_dists * ((H + W) / 2), dim=-1) / tau  # [B,1,H,W]

    # 5. Penalty: how much closer than desired_min_dist each pixel is
    proximity = F.relu(desired_min_dist - d_closest_obstacle)  # [B,1,H,W]

    # Normalize proximity and weight by path prediction confidence
    proximity_norm = proximity / desired_min_dist
    penalty = proximity_norm * path_pred  # [B,1,H,W]

    # Mean penalty scaled by number of path pixels (pixel_sums: [B])
    mean_penalty = penalty.mean(dim=(2, 3)) * pixel_sums.view(-1, 1)  # [B,1]
    max_penalty  = penalty.amax(dim=(2, 3))  # [B,1]

    return -mean_penalty, -max_penalty


def compute_path_cost_approximation(
    path: torch.Tensor,
    eps: float = 1e-8,
    use_uniform_step_cost: bool = False,
):

    """

    Parameters
    - path: torch.Tensor, shape [B,1,H,W]
        Soft path predictions indicating where the agent is predicted to travel.
    - eps: float
        Small epsilon to avoid division by zero.

    Returns
    - aprox_path_pixel_cost: torch.Tensor, shape [B,1,H,W]
        Per-pixel approximated traversal cost (clipped between 0 and sqrt(2)). This approximates
        path length contributions by weighting neighbors according to their connectivity
        (4-neighbors and diagonal neighbors have different costs).

    Short description
    - Uses a 3x3 convolution with a kernel encoding approximate edge lengths to compute a
      local path-length approximation around each pixel, normalized by the local mean
      path confidence in the 3x3 neighborhood.

    Detailed steps
    1. Build a cost kernel where orthogonal neighbors cost 1 and diagonal neighbors cost sqrt(2).
       The kernel is placed in a 3x3 matrix and reshaped to [1,1,3,3] for conv2d.
    2. Convolve path with this kernel to compute a weighted sum of neighbor contributions (aprox_cost).
       Also convolve with a ones kernel to get the local sum of prediction mass (mean_path_predictions_in_neighborhood).
    3. Normalize the cost by the local mass to get an approximate per-pixel cost and multiply by the pixel
       confidence to zero out non-path pixels.
    4. Clamp the resulting per-pixel cost to be within [0, sqrt(2)].

    Notes:
    - This is a local approximation and not a true path-length. It is intended to be cheap and differentiable.

    """

    
    sqrt2 = 2 ** (1/2)
    ones_kernel = torch.ones(size=(1, 1, 3, 3), dtype=torch.float32, device=path.device)
    
    if use_uniform_step_cost:
        cost_kernel = torch.ones(size=(1, 1, 3, 3), dtype=torch.float32, device=path.device)
    else:
        cost_kernel = torch.tensor([[sqrt2, 1.0, sqrt2], [1.0, 1.0, 1.0], [sqrt2, 1.0, sqrt2]], device=path.device).view(1, 1, 3, 3)  # one in the middle to make normalization consistent

    # Convolution approximating cost contributions from neighbors
    aprox_cost = F.conv2d(
        input=path,
        weight=cost_kernel,
        bias=None,
        stride=1,
        padding=1
    )

    # Local count of predictions in 3x3 neighborhood
    mean_path_predictions_in_neighborhood = F.conv2d(
        input=path,
        weight=ones_kernel,
        bias=None,
        stride=1,
        padding=1
    )

    # Normalize and scale; multiply by path to keep only path pixel costs
    aprox_path_pixel_cost = torch.clamp(
        input=(aprox_cost / (mean_path_predictions_in_neighborhood + eps)) * path, 
        min=0.0,
        max=sqrt2
    )

    return aprox_path_pixel_cost

def exp_ease_in(x: torch.Tensor, a=4):
    return (1 - torch.exp(-a*(x**2))) / 1 - math.exp(-a)


def soft_optimal_connectivity(
    path_pred: torch.Tensor,
    start: torch.Tensor, 
    end: torch.Tensor, 
    obstacles: torch.Tensor,
    steps: int = 125, 
    sharpness_start: int = 8.0,
    sharpness_end: int = 16.0,
    use_uniform_step_cost: bool = False,
):
    B, _, H, W = path_pred.shape

    # Extract start and end coordinates using argmax along spatial dimensions
    # For each batch, find the location of maximum value in the start and end masks
    start_flat = start.view(B, -1)  # [B, H*W]
    end_flat = end.view(B, -1)      # [B, H*W]

    start_idx = torch.argmax(start_flat, dim=1)  # [B] flattened indices
    end_idx = torch.argmax(end_flat, dim=1)      # [B]

    # Convert flattened indices back to 2D coordinates (x, y)
    start_x = start_idx % W  # [B] x-coordinate
    start_y = start_idx // W  # [B] y-coordinate
    end_x = end_idx % W      # [B]
    end_y = end_idx // W     # [B]

    # Initialize seeds with zeros
    seeds = torch.full(
        size=path_pred.shape,  # [B,1,H,W]
        device=path_pred.device,
        dtype=torch.float32,
        fill_value=1e6
    )

    path_pred_with_end_included = torch.clone(path_pred)  # [B,1,H,W]

    # Place a seed at the start point (use the batch indices and the coordinates)
    seeds[torch.arange(B), 0, start_y, start_x] = 0.0

    reachable = seeds.clone()  # [B,1,H,W]
    
    B, C, H, W = reachable.shape

    sqrt2 = math.sqrt(2)
    if use_uniform_step_cost:
        w = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.1, 1.0], [1.0, 1.0, 1.0]], device=path_pred.device).view(1, 1, 9, 1)
    else:
        # default, use sqrt(2) for diagonals
        w = torch.tensor([[sqrt2, 1.0, sqrt2], [1.0, 0.1, 1.0], [sqrt2, 1.0, sqrt2]], device=path_pred.device).view(1, 1, 9, 1)

    sharpness = torch.linspace(
        sharpness_start, sharpness_end, steps, device=path_pred.device
    )

    inv = 1 - path_pred_with_end_included

    # --- Main propagation loop ---
    for t in range(steps):
        tau = max(1.0 / sharpness[t], 0.05)

        reachable_padded = F.pad(
            reachable,
            pad=(1, 1, 1, 1),  # (left, right, top, bottom)
            mode="constant",
            value=1e6
        )
        patches = F.unfold(reachable_padded, kernel_size=3, padding=0)
        patches = patches.view(B, C, 9, H*W)

        weighted = patches + w

        out = -tau * torch.logsumexp(-weighted / tau, dim=2)  # torch.amin(weighted, dim=2)  

        reachable = out.view(B, C, H, W)

        reachable = reachable + inv + obstacles * 1e6

    # Read connectivity at the end point (extract the value at the end coordinates)
    end_vals = reachable[torch.arange(B), 0, end_y, end_x]  # [B]

    cost_threshold = sqrt2 * steps + steps + 1  # sqrt2 * steps + steps is maximum cost archievable while still connecting start and end

    # if path signal cannot reach goal (=highly unconnected path) the cost accumulated at the goal pixel will come from the 1e6 initalization of seeds
    # -> in that case the gradient is garbage
    return  -torch.clamp_max(end_vals, max=cost_threshold)  # kill gradients if acumulated cost is larger than possible 



def soft_optimal_connectivity_tunable(
    path_pred: torch.Tensor,
    start: torch.Tensor, 
    end: torch.Tensor, 
    obstacles: torch.Tensor,
    steps: int = 125, 
    sharpness_start: int = 8.0,
    sharpness_end: int = 16.0,
    optimality_factor: float = 1.0,
):
    B, _, H, W = path_pred.shape

    # Extract start and end coordinates using argmax along spatial dimensions
    # For each batch, find the location of maximum value in the start and end masks
    start_flat = start.view(B, -1)  # [B, H*W]
    end_flat = end.view(B, -1)      # [B, H*W]

    start_idx = torch.argmax(start_flat, dim=1)  # [B] flattened indices
    end_idx = torch.argmax(end_flat, dim=1)      # [B]

    # Convert flattened indices back to 2D coordinates (x, y)
    start_x = start_idx % W  # [B] x-coordinate
    start_y = start_idx // W  # [B] y-coordinate
    end_x = end_idx % W      # [B]
    end_y = end_idx // W     # [B]

    # Initialize seeds with zeros
    seeds = torch.full(
        size=path_pred.shape,  # [B,1,H,W]
        device=path_pred.device,
        dtype=torch.float32,
        fill_value=1e6
    )

    path_pred_with_end_included = torch.clone(path_pred)  # [B,1,H,W]

    # Place a seed at the start point (use the batch indices and the coordinates)
    seeds[torch.arange(B), 0, start_y, start_x] = 0.0

    reachable = seeds.clone()  # [B,1,H,W]
    
    B, C, H, W = reachable.shape

    sqrt2 = math.sqrt(2)
    w = torch.tensor([[sqrt2, 1.0, sqrt2], [1.0, 0.1, 1.0], [sqrt2, 1.0, sqrt2]], device=path_pred.device).view(1, 1, 9, 1)

    w = optimality_factor * w  # trade off optimality vs mission objectives

    sharpness = torch.linspace(
        sharpness_start, sharpness_end, steps, device=path_pred.device
    )

    inv = 1 - path_pred_with_end_included

    # --- Main propagation loop ---
    for t in range(steps):
        tau = max(1.0 / sharpness[t], 0.05)

        reachable_padded = F.pad(
            reachable,
            pad=(1, 1, 1, 1),  # (left, right, top, bottom)
            mode="constant",
            value=1e6
        )
        patches = F.unfold(reachable_padded, kernel_size=3, padding=0)
        patches = patches.view(B, C, 9, H*W)

        weighted = patches + w

        out = -tau * torch.logsumexp(-weighted / tau, dim=2)  # torch.amin(weighted, dim=2)  

        reachable = out.view(B, C, H, W)

        reachable = reachable + inv + obstacles * 1e6

    # Read connectivity at the end point (extract the value at the end coordinates)
    end_vals = reachable[torch.arange(B), 0, end_y, end_x]  # [B]

    cost_threshold = sqrt2 * steps + steps + 1  # sqrt2 * steps + steps is maximum cost archievable while still connecting start and end

    # if path signal cannot reach goal (=highly unconnected path) the cost accumulated at the goal pixel will come from the 1e6 initalization of seeds
    # -> in that case the gradient is garbage
    return  -torch.clamp_max(end_vals, max=cost_threshold)  # kill gradients if acumulated cost is larger than possible 
