import torch
import torch.nn.functional as F


@torch.compile
def compute_gradual_std_mask_3D(
    logits: torch.Tensor,
    state: torch.Tensor,
    add_pixel_ratio: float,
    amount_of_pixels: int = 2
):
    """
    Parameters:
    - logits: Tensor of shape [B, 1, D, H, W]
    - state: Tensor of shape [B, D, H, W]

    """

    # calculate base path
    base_path = F.tanh(logits) >= 0.0

    # calculate obstacle map
    obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=state.device)
    obstacle_map = torch.all(state == obstacle_color[None, :, None, None, None], axis=1)

    log_stds = torch.full(
        size=logits.shape,
        fill_value=-10.0,  # -> std will then be in 10^-5 range -> effectively zero
        device=logits.device
    ) # shape [B, 1, H, W]

    # calculate immediate neighbor region, required either way
    neighbor_region = compute_immediate_neighbor_region_3D(path=base_path)

    if torch.rand((), device=logits.device) <= add_pixel_ratio:
        # we do not want to explore on obstacles -> zero out obstacles from the region that is sampled from
        neighbor_region_without_obstacles = torch.logical_and(
            input=neighbor_region,
            other=~obstacle_map
        )

        # add pixels
        log_stds = sample_random_3D(
            region=neighbor_region_without_obstacles,
            amount=amount_of_pixels,
            value=10.0,  # -> std will then be in 10^5 range -> makes the probability of this pixel being sampled as part of the path effectively 50/50
            out=log_stds
        )

    else:
        # remove pixels
        # calculate path border (included in path)
        boundary_region = compute_path_boundary(
            path_neighbors=neighbor_region,
            path=base_path
        )

        # remove pixels
        log_stds = sample_random_3D(
            region=boundary_region,
            amount=amount_of_pixels,
            value=10.0,  # -> std will then be in 10^5 range -> makes the probability of this pixel being sampled as part of the path effectively 50/50
            out=log_stds
        )

    return log_stds


def sample_random_3D(
    region: torch.Tensor,
    amount: int,
    value: float,
    out: torch.Tensor
):
    B, _, D, H, W = region.shape

    for i in range(B):
        candidates = region[i][0].nonzero()  # [B, 3]
        random_idxs = torch.randperm(len(candidates))

        for j in range(min(amount, len(random_idxs))):
            x, y, z = candidates[random_idxs[j]]
            out[i, 0, x, y, z] = value

    return out


def compute_immediate_neighbor_region_3D(
    path: torch.Tensor
):
    
    """
    Returns a map with only the immediate neighbor pixels of the path (=every pixel next to a path pixel which is not part of the path itself)

    Parameters:
    - path: Tensor with 1: path, 0: no path, shape=[B, 1, D, H, W]

    Returns:
    - described path map

    """

    kernel = torch.full(
        size=(1, 1, 3, 3, 3), 
        fill_value=1.0,
        device=path.device,
        dtype=torch.float32
    )

    # Apply 2d convolution with proximity kernel
    # >= 1 for pixels belonging to path or being immediate neighbors
    path_with_neighbor_region_inconsistent = F.conv3d(
        input=path.to(dtype=torch.float32),
        weight=kernel,
        padding=1,  # such that input shape = result shape
        bias=None
    )  

    # clip upper bound such that 1: union(path, immediate neighbors) 0: everything else
    path_with_neighbor_region = path_with_neighbor_region_inconsistent >= 1.0

    # mask out path
    neighbor_region = torch.logical_and(path_with_neighbor_region, ~path)

    return neighbor_region
    

def compute_path_boundary(
    path_neighbors: torch.Tensor,
    path: torch.Tensor
):
    
    """
    Returns a map with only the boundary pixels of the path. Effectively calculates the neighbors of only the path neighbors (those not included in path) and selects intersection with path. Should be somewhat efficient on GPUs.

    Parameters:
    - path_neighbors: Tensor with 1: neighbor, 0: no neighbor, shape=[B, 1, D, H, W]
    - path: Tensor with 1: path, 0: no path, shape=[B, D, H, W]

    Returns:
    - described path map

    """

    kernel = torch.full(
        size=(1, 1, 3, 3, 3),
        fill_value=1.0,
        device=path.device,
        dtype=torch.float32
    )

    # Apply 3d convolution with proximity kernel
    # >= 1 for pixels belonging to neighbors or being immediate neighbors of neighbors
    neighbors_with_neighbor_region_inconsistent = F.conv3d(
        input=path_neighbors.to(dtype=torch.float32),
        weight=kernel,
        padding=1,  # such that input shape = result shape
        bias=None
    )  

    # clip upper bound such that 1: union(path, immediate neighbors) 0: everything else
    neighbors_with_neighbor_region = neighbors_with_neighbor_region_inconsistent >= 1.0

    # mask out path
    path_boundary = torch.logical_and(neighbors_with_neighbor_region, path)

    return path_boundary
    