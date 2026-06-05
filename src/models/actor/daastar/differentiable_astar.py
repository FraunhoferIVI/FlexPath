# -------------------------------------------------------------------------------------------------------------
# File: differentiable_astar.py
# Project: DAA*: Deep Angular A Star for Image-based Path Planning
# Contributors:
#     Zhiwei Xu <zwxu064@gmail.com>
#
# Copyright (c) 2025 Zhiwei Xu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# -------------------------------------------------------------------------------------------------------------

import math
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from typing import List, NamedTuple, Optional


@torch.no_grad()
def cal_path_angle(path_stack):
    path_stack = path_stack.double()
    num_nodes = path_stack.shape[-1]
    num_samples = path_stack.shape[0]
    path_angles = torch.zeros(num_samples, dtype=torch.float64, device=path_stack.device)

    for idx in range(num_nodes):
        # Only for every 3 nodes, forming 2 vectors to calculate the angle
        if idx + 2 >= num_nodes: continue

        current_loc = path_stack[:, :, idx]
        next_loc = path_stack[:, :, idx + 1]
        next_next_loc = path_stack[:, :, idx + 2]

        # Only when the very remote one is valid, the vectors will be valid
        # For some batches, the path is shorter and the angle caculation will terminate earlier
        valid_indices = (next_next_loc[:, 0] >= 0) * (next_next_loc[:, 1] >= 0)

        if valid_indices.float().sum() == 0:
            break
        else:
            v_1 = (next_loc - current_loc)[valid_indices]
            v_2 = (next_next_loc - next_loc)[valid_indices]

            # To get the angle using unit vectors
            v_1_normed = v_1 / v_1.pow(2).sum(1, keepdim=True).sqrt()
            v_2_normed = v_2 / v_2.pow(2).sum(1, keepdim=True).sqrt()

            cos_v = torch.einsum('bn,bn->b', v_1_normed, v_2_normed)
            angles = torch.arccos(cos_v) * 180 / torch.pi
            path_angles[valid_indices] += angles

    return path_angles

def compute_step_cost(
    selected_node_maps: torch.Tensor,
    cost_filter: torch.Tensor
):
    """
    Sets each neighbor of a selected node to the corresponding step cost

    Args:
        selected_node_maps (torch.Tensor): [B, H, W] one-hot tensor for selected node
        cost_filter (torch.Tensor): [1, 1, 3, 3] kernel of step cost
    Returns:
        torch.Tensor: step cost
    """

    x = selected_node_maps.unsqueeze(1)
    y = F.conv2d(
        x,
        weight=cost_filter,
        padding=1
    )

    return y.squeeze(1)

# =============================================================================================================

class AstarOutput(NamedTuple):
    histories: torch.tensor
    paths: torch.tensor
    intermediate_results: Optional[List[dict]] = None
    cost_maps: torch.tensor = None
    obstacles_maps: torch.tensor = None
    heuristic: torch.tensor = None
    path_stack: torch.tensor = None
    path_angles: torch.tensor = None
    path_costs: torch.tensor = None
    hist_coordinates: torch.tensor = None
    history_probs: torch.tensor = None

# =============================================================================================================

# def get_heuristic(
#     goal_maps,
#     tb_factor=0.001
# ):
#     # some preprocessings to deal with mini-batches
#     num_samples, H, W = goal_maps.shape[0], goal_maps.shape[-2], goal_maps.shape[-1]
#     grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
#     loc = torch.stack(grid, dim=0).type_as(goal_maps)
#     loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
#     goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
#     goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)

#     # chebyshev distance
#     dxdy = torch.abs(loc_expand - goal_loc_expand)

#     # NOTE: this is just dxdy.max(dim=1)[0], defined in wiki of Chebyshev distance max(|x-y|)
#     h = dxdy.sum(dim=1) - dxdy.min(dim=1)[0]

#     euc = torch.sqrt(((loc_expand - goal_loc_expand) ** 2).sum(1))
#     h = (h + tb_factor * euc).reshape_as(goal_maps)

#     return h

# octile
def get_heuristic(goal_maps: torch.tensor, tb_factor: float = 0.001) -> torch.tensor:
    """
    Get heuristic function for A* search (chebyshev + small const * euclidean)

    Args:
        goal_maps (torch.tensor): one-hot matrices of goal locations
        tb_factor (float, optional): small constant weight for tie-breaking. Defaults to 0.001.

    Returns:
        torch.tensor: heuristic function matrices
    """

    # some preprocessings to deal with mini-batches
    num_samples, H, W = goal_maps.shape[0], goal_maps.shape[-2], goal_maps.shape[-1]
    grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    loc = torch.stack(grid, dim=0).type_as(goal_maps)
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)

    # switch to octile distance for non-uniform step costs (sqrt(2) for diagonal ones)
    dxdy = torch.abs(loc_expand - goal_loc_expand)
    h = dxdy.min(dim=1)[0] * (2**0.5) + torch.abs(dxdy[:, 0] - dxdy[:, 1])
    euc = torch.sqrt(((loc_expand - goal_loc_expand) ** 2).sum(1))
    h = (h + tb_factor * euc).reshape_as(goal_maps)

    return h

# =============================================================================================================

def _st_softmax_noexp(
    val,
    return_onehot_prob=False,
    mode='max',
    eps: float = 1e-8
):
    batch_size = val.shape[0]
    val_ = val.reshape(batch_size, -1)
    y = val_ / (val_.sum(dim=-1, keepdim=True) + eps)  # added eps for numerical stability

    if mode.find('randomwalk') > -1:
        if mode.find('_') > -1:
            topk = int(mode.split('_')[-1])
            topk_threshold = torch.topk(y, topk, dim=1)[0][:, -1:]

            # It is very likely (say top-1) that many nodes have the same probability,
            # then the index is randomly sampled by multinomial, but this is fine
            y_filted = y * (y >= topk_threshold)
        else:
            y_filted = y

        ind = torch.multinomial(y_filted, 1).view(batch_size)
    else:
        _, ind = y.max(dim=-1)  # MOD

    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)

    # NOTE
    diff_onehot_index = (y_hard - y).detach() + y

    if not return_onehot_prob and not return_onehot_prob:
        return diff_onehot_index
    else:
        result_dict = {'onehot_index': diff_onehot_index}

        if return_onehot_prob:
            result_dict.update({'onehot_prob': y * y_hard.detach()})

        return result_dict

# =============================================================================================================

def expand(
    x,
    neighbor_filter
):
    # NOTE x: (1, batch, h, w), neighbour: (batch, 1, 3, 3)
    # x has only a single 1, using a 3*3 kernel to find 9 nodes in this kernel,
    # F.conv2d operation is x*w^T
    if True:
        batch, h, w = x.shape
        kernel_c, kernel_h, kernel_w = neighbor_filter.shape[-3:]
        y = F.conv2d(
            x.view(batch, 1, h, w),
            neighbor_filter[0:1].view(1, kernel_c, kernel_h, kernel_w),
            padding=1
        )
        y = y.view(batch, h, w)
    else:
        x = x.unsqueeze(0)
        num_samples = x.shape[1]
        y = F.conv2d(x, neighbor_filter, padding=1, groups=num_samples).squeeze()
        y = y.squeeze(0)

    return y

# =============================================================================================================

def backtrack(
    start_maps,
    goal_maps,
    parents,
    current_t,
    enable_path_stack
):
    num_samples, height, width = start_maps.shape
    parents = parents.type(torch.long)
    goal_maps = goal_maps.type(torch.long)
    start_maps = start_maps.type(torch.long)
    path_maps = goal_maps.type(torch.long)
    num_samples = len(parents)
    loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
    batch_indices = torch.arange(num_samples, dtype=torch.long, device=parents.device)

    if enable_path_stack:
        path_stack = -torch.ones(
            (num_samples, 2, height * width),
            dtype=torch.float32,
            device=goal_maps.device
        )
        end_point = torch.nonzero(goal_maps.view(num_samples, height, width) == 1)
        path_stack[batch_indices, :, 0] = end_point[:, 1:].float()
    else:
        path_stack = None

    if False:
        for idx in range(current_t):
            path_maps.view(num_samples, -1)[range(num_samples), loc] = 1

            if enable_path_stack:
                path_stack[range(num_samples), 0, idx] = (loc / width).floor().float()
                path_stack[range(num_samples), 1, idx] = (loc % width).float()

            loc = parents[range(num_samples), loc]
    else:
        for idx in range(current_t):
            valid_indices = (loc >= 0)

            if valid_indices.float().sum() == 0: continue

            batch_indices = batch_indices[valid_indices]
            loc = loc[valid_indices]
            path_maps.view(num_samples, -1)[batch_indices, loc] = 1

            if enable_path_stack:
                path_stack[batch_indices, 0, idx + 1] = (loc / width).floor().float()
                path_stack[batch_indices, 1, idx + 1] = (loc % width).float()

            loc = parents[batch_indices, loc]

    if enable_path_stack:
        return path_stack, path_maps
    else:
        return path_maps

# =============================================================================================================

def get_h_w(data):
    width = data.shape[-1]
    h, w = (data / width).floor().float(), (data % width).float()

    return h, w

# =============================================================================================================

def normalize_vector(
    vector_h,
    vector_w
):
    # dist = (vector_h ** 2 + vector_w ** 2).sqrt()
    dist = torch.sqrt(torch.pow(vector_h, 2) + torch.pow(vector_w, 2))
    dist[dist == 0] = 1

    return vector_h / dist, vector_w / dist

# =============================================================================================================

def cal_step_angle(
    selected_nodes,
    parents,
    neighbor_nodes,
    neighbor_filter,
    index_maps=None
):
    num_samples, height, width = selected_nodes.shape
    parents = parents.view(num_samples, height, width)

    if index_maps is None:
        index_maps = torch.arange(
            height * width, dtype=torch.long, device=selected_nodes.device)
        index_maps = index_maps.view(1, height, width)

    # Find the index corresponding to the binary nodes
    selected_node_index = selected_nodes * index_maps
    neighbor_node_index = neighbor_nodes * index_maps
    selected_node_parent_index = selected_nodes * parents

    # Expand one node to all its neighbours
    selected_node_index = expand(selected_node_index, neighbor_filter)
    selected_node_parent_index = expand(selected_node_parent_index, neighbor_filter)

    selected_node_h, selected_node_w = get_h_w(selected_node_index)
    neighbor_node_h, neighbor_node_w = get_h_w(neighbor_node_index)
    selected_node_parent_h, selected_node_parent_w = get_h_w(selected_node_parent_index)

    vector_1_h = neighbor_node_h - selected_node_h
    vector_1_w = neighbor_node_w - selected_node_w
    vector_2_h = selected_node_h - selected_node_parent_h
    vector_2_w = selected_node_w - selected_node_parent_w

    vector_1_h, vector_1_w = normalize_vector(vector_1_h, vector_1_w)
    vector_2_h, vector_2_w = normalize_vector(vector_2_h, vector_2_w)

    angles = torch.arccos(vector_1_h * vector_2_h + vector_1_w * vector_2_w)

    # The current nodes have no angle, only the neighbours have, *(1 - selected_nodes);
    # The history nodes have no angle because they will not be neighbours, *open_maps
    # If the parent of a current node is invalid, this batch has no angle, *(parents >= 0) > 0 with batch wise.
    # angles = angles * neighbor_nodes * (1 - history) * ((parents >= 0).sum((1, 2), keepdim=True) > 0).float()

    return angles

# =============================================================================================================

class DifferentiableAstar(nn.Module):
    def __init__(
        self,
        config,
        disable_heuristic=False
    ):
        super().__init__()

        neighbor_filter = torch.ones(1, 1, 3, 3)
        neighbor_filter[0, 0, 1, 1] = 0

        if config.num_dirs == 4:
            neighbor_filter[0, 0, 0, 0] = 0
            neighbor_filter[0, 0, 0, 2] = 0
            neighbor_filter[0, 0, 2, 0] = 0
            neighbor_filter[0, 0, 2, 2] = 0

        # init cost filter
        sqrt2 = 2 ** 0.5
        self.cost_filter = nn.Parameter(
            data=torch.tensor([[sqrt2, 1.0, sqrt2], [1.0, 0.0, 1.0], [sqrt2, 1.0, sqrt2]]).view(1, 1, 3, 3),
            requires_grad=False
        )

        self.neighbor_filter = nn.Parameter(neighbor_filter, requires_grad=False)
        self.get_heuristic = get_heuristic
        self.config = config
        assert (config.Tmax > 0) & (config.Tmax <= 1), "Tmax must be within (0, 1]"
        self.index_maps = None
        self.disable_heuristic = disable_heuristic
        self.init_path_angles = torch.tensor([-1], dtype=torch.float32)

        
        # self.config.enable_angle = False
        self.infer_optimality = False


        # NOTE: have not yet checked, just theoretically it is
        # Rotation angle, a large angle constraint leads to large histories
        if self.config.enable_train_rotation_const:
            self.rotation_const_param = torch.nn.Parameter(
                torch.tensor([0], dtype=torch.float32),
                requires_grad=True
            )
            self.rotation_const_act = torch.nn.Sigmoid()
        else:
            self.rotation_const_param = None
            self.rotation_const_act = None

        # Important!!!
        # G_ratio, initial 10 to get a large g_ratio (sigmoid(10)=1) which will lead to a large histories to be learned;
        # That is from a large g_ratio (close to Dijkstra) to a small g_ratio (faster with small histories)
        if self.config.enable_train_g_ratio:
            self.g_ratio_param = torch.nn.Parameter(
                torch.tensor([0], dtype=torch.float32),
                requires_grad=True
            )
            self.g_ratio_act = torch.nn.Sigmoid()
        else:
            self.g_ratio_param = None
            self.g_ratio_act = None

        # Apply automatic angle maximization and minimization
        self.rotation_weight_param = torch.nn.Parameter(
            torch.tensor([0], dtype=torch.float32),
            requires_grad=True
        )
        self.rotation_weight_act = torch.nn.Sigmoid()

        current_g_ratio = self.get_g_ratio()

        if isinstance(current_g_ratio, torch.Tensor):
            current_g_ratio = current_g_ratio.item()

        current_rotation_const = self.get_rotation_const()

        if isinstance(current_rotation_const, torch.Tensor):
            current_rotation_const = current_rotation_const.item()

    def get_rotation_const(self):
        if self.infer_optimality:
            return torch.tensor(0.978)
        if self.rotation_const_act is None:
            # 2023.12.11 when rotation.const is torch.Tensor or a const, there would be round error
            if not isinstance(self.config.rotation.const, torch.Tensor):
                self.config.rotation.const = torch.tensor(self.config.rotation.const, dtype=torch.float32)

            return self.config.rotation.const
        else:
            return self.rotation_const_act(self.rotation_const_param)

    def get_rotation_weight(self):
        if self.infer_optimality:
            return torch.tensor(0.964)  # value from paper
        return self.rotation_weight_act(self.rotation_weight_param)

    def get_g_ratio(self):
        if self.infer_optimality:
            return torch.tensor(0.13)  # the learned value favors too many expansions
        if self.g_ratio_act is None:
            # 2023.12.11 when g_ratio is torch.Tensor or a const, there would be round error
            if not isinstance(self.config.g_ratio, torch.Tensor):
                self.config.g_ratio = torch.tensor(self.config.g_ratio, dtype=torch.float32)

            return self.config.g_ratio
        else:
            return self.g_ratio_act(self.g_ratio_param)

    def get_path_cost(self, cost_maps, path_maps, path_angles, heuristic):
        # This actual cost consists of heuristic, cost_maps (or say number of steps without learning), and angles.
        g_ratio = self.get_g_ratio()
        rotation_const = self.get_rotation_const()
        num_samples = cost_maps.shape[0]

        if True:
            path_costs = ((cost_maps + (1 - g_ratio) * heuristic) * path_maps).view(num_samples, -1).sum(1) \
                + g_ratio * rotation_const * path_angles * torch.pi / 180
        else:  # keep unary, and change g_ratio for message passing
            path_costs = (((1 + g_ratio) * cost_maps + heuristic) * path_maps).view(num_samples, -1).sum(1) \
                + g_ratio * rotation_const * path_angles * torch.pi / 180

        return path_costs

    def forward(
        self,
        cost_maps: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        obstacles_maps: torch.tensor,
        store_intermediate_results: bool = False,
        store_hist_coordinates: bool = False,
        disable_heuristic: bool = False,
        disable_compute_path_angle: bool = False
    ):
        assert store_intermediate_results == False
        assert cost_maps.ndim == 4
        assert start_maps.ndim == 4
        assert goal_maps.ndim == 4
        assert obstacles_maps.ndim == 4

        obstacles_maps = -obstacles_maps + 1  # this code uses a very weird occupancy format

        # from PIL import Image
        # import numpy as np

        # Image.fromarray(((cost_maps[0, 0] / cost_maps[0, 0].max()).cpu().detach().numpy() * 255).astype(np.uint8)).save("img3.png")
        # Image.fromarray((obstacles_maps[0, 0].cpu().detach().numpy() * 255).astype(np.uint8)).save("obst3.png")

        # print("Rotation param: ", self.rotation_const_param, self.get_g_ratio(), self.get_rotation_weight())

        # cost_maps = cost_maps * 10

        # print("Obstacle maps at start", obstacles_maps[start_maps.bool()])

        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]

        # NOTE: when obstacles_maps is 1 means it can be a neighbour; 0 means cannot be.
        # This is only useful for explicitly marked obstacles images NOT warcraft images.
        obstacles_maps = obstacles_maps[:, 0]
        num_samples, height, width = start_maps.shape

        if True:  # NOTE do not broadcast batches to filter, just use (1, 1, 3, 3)
            neighbor_filter = self.neighbor_filter
        else:
            neighbor_filter = self.neighbor_filter
            neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples, 0)

        size = start_maps.shape[-1]

        open_maps = start_maps
        histories = torch.zeros_like(start_maps)
        history_probs = torch.zeros_like(start_maps)
        intermediate_results = []

        heuristic = self.get_heuristic(goal_maps, self.config.tb_factor)
        if disable_heuristic: heuristic = heuristic * 0  # switch to Dijkstra
        heuristic = heuristic * (1 - start_maps) * (1 - goal_maps)

        # print("\n\nNorm heuristic: ", self.config.heuristic.enable_norm_heuristic, self.config.heuristic.const)

        if self.config.heuristic.enable_norm_heuristic:
            heuristic = heuristic / heuristic.view(num_samples, -1).max(dim=1)[0].view(num_samples, 1, 1)
            heuristic = heuristic * self.config.heuristic.const

        h = heuristic + cost_maps
        g = torch.zeros_like(start_maps)
        parents = -torch.ones_like(start_maps).reshape(num_samples, -1)

        size = cost_maps.shape[-1]
        Tmax = self.config.Tmax if self.training else 1.0
        Tmax = int(Tmax * size * size)
        hist_coordinates = [] if store_hist_coordinates else None

        g_ratio = self.get_g_ratio()
        rotation_const = self.get_rotation_const()
        rotation_weight = self.get_rotation_weight()

        # print(g_ratio, rotation_const, rotation_weight, self.config.enable_inv_rotation)

        if self.config.enable_angle and (self.index_maps is None):
            self.index_maps = torch.arange(height * width, dtype=torch.long, device=start_maps.device)
            self.index_maps = self.index_maps.view(1, height, width)

        

        # print(cost_maps.min(), cost_maps.max(), cost_maps.mean(), cost_maps.std())
        # print("Gratio: ", g_ratio)
        for t in range(Tmax):
            g, h, open_maps, histories, history_probs, parents, is_unsolved = step(g, h, g_ratio, cost_maps, goal_maps, neighbor_filter, obstacles_maps, num_samples, height, width, self.config.enable_angle, self.config.expand_mode, self.index_maps, self.config.enable_inv_rotation, rotation_const, rotation_weight, open_maps, history_probs, histories, parents, self.cost_filter)
            # Image.fromarray(((histories[0]).cpu().detach().numpy() * 255).astype(np.uint8)).save(f"expansions/{t}.png")
            if torch.all(is_unsolved.flatten() == 0): break

        # Store the coordinates for every step with a single open node
        if store_hist_coordinates: hist_coordinates = torch.stack(hist_coordinates, 1)

        # Backtracking
        path_stack, path_maps = backtrack(start_maps, goal_maps, parents, t, enable_path_stack=True)

        # Calculate angles
        # if disable_compute_path_angle:
        #     if self.init_path_angles.device != start_maps.device:
        #         self.init_path_angles = self.init_path_angles.to(start_maps.device)

        #     path_angles = self.init_path_angles.repeat(num_samples)
        # else:
        #     path_angles = cal_path_angle(path_stack)
        path_angles = None

        # The optimization cost
        # g * goal_maps == (cost_maps * path_maps).view(num_samples, -1).sum(1) + path_angles * torch.pi / 180
        # over the binary path, but does not consider heuristic which is contained in the true optimization cost.
        # So, it should be, the following three are the same
        # path_costs = ((cost_maps + (1 - self.g_ratio) * heuristic) * path_maps).view(num_samples, -1).sum(1) + self.g_ratio * path_angles * torch.pi / 180
        # path_costs = (self.g_ratio * g * goal_maps + (1 - self.g_ratio) * h * path_maps).view(num_samples, -1).sum(1)
        # path_costs = self.get_path_cost(cost_maps, path_maps, path_angles, heuristic)

        if self.config.dataset_name in ['warcraft', 'pkmn']:
            path_costs = (cost_maps * path_maps).view(num_samples, -1).sum(1) / self.config.encoder.const
        else:
            path_costs = -1

        if store_intermediate_results:
            intermediate_results.append(
                {
                    "histories": histories.unsqueeze(1).detach(),
                    "paths": path_maps.unsqueeze(1).detach(),
                }
            )

        # from PIL import Image
        # import numpy as np

        # for i in range(len(path_maps)):
        #     Image.fromarray((path_maps[i].cpu().detach().numpy() * 255).astype(np.uint8)).save(f"pathsdaa/path{i}.png")

        return AstarOutput(
            histories.unsqueeze(1),
            path_maps.unsqueeze(1),
            intermediate_results,
            cost_maps.unsqueeze(1),
            obstacles_maps.unsqueeze(1),
            heuristic.unsqueeze(1),
            path_stack,
            path_angles,
            path_costs,
            hist_coordinates,
            history_probs.unsqueeze(1)
        )
    

@torch.compile
def step(g, h, g_ratio, cost_maps, goal_maps, neighbor_filter, obstacles_maps, num_samples, height, width, enable_angle, expand_mode, index_maps, enable_inv_rotation, rotation_const, rotation_weight, open_maps, history_probs, histories, parents, cost_filter):
    f = g_ratio * g + (1 - g_ratio) * h
    f_exp = torch.exp(-1 * f / math.sqrt(cost_maps.shape[-1]))
    f_exp = f_exp * open_maps
    result_dict = _st_softmax_noexp(
        f_exp,
        return_onehot_prob=True,
        mode=expand_mode
    )
    selected_node_maps = result_dict['onehot_index']
    selected_node_probs = result_dict['onehot_prob']

    history_probs = (1 - selected_node_maps.detach()) * history_probs \
        + selected_node_maps.detach() * selected_node_probs

    # break if arriving at the goal
    dist_to_goal = (selected_node_maps * goal_maps).sum((1, 2), keepdim=True)
    is_unsolved = (dist_to_goal < 1e-8).float()

    histories = histories + selected_node_maps
    histories = torch.clamp(histories, 0, 1)
    open_maps = open_maps - is_unsolved * selected_node_maps
    open_maps = torch.clamp(open_maps, 0, 1)

    # open neighboring nodes, add them to the openlist if they satisfy certain requirements
    neighbor_nodes = expand(selected_node_maps, neighbor_filter)
    neighbor_nodes = neighbor_nodes * obstacles_maps

    # update g if one of the following conditions is met
    # 1) neighbor is not in the close list (1 - histories) nor in the open list (1 - open_maps)
    # 2) neighbor is in the open list but g < g2
    # NOTE: cost_maps here will not be differentiable
    g2 = expand((g + cost_maps) * selected_node_maps, neighbor_filter) + compute_step_cost(selected_node_maps, cost_filter)  # MODIFIED: add step cost here
    g2 = g2.detach()

    # Add rotation angle here
    if enable_angle:
        angles = cal_step_angle(
            selected_node_maps,
            parents,
            neighbor_nodes,
            neighbor_filter,
            index_maps=index_maps
        )
        angles = angles * neighbor_nodes * (1 - histories) \
            * ((parents.view(num_samples, height, width) >= 0).sum((1, 2), keepdim=True) > 0).float()
        angles = angles.detach()

        # Apply automatic angle maximization and minimization
        if enable_inv_rotation is None:
            angle_cost = rotation_const * \
                (rotation_weight * angles + (1 - rotation_weight) * (torch.pi - angles))
        else:
            if enable_inv_rotation:
                angle_cost = rotation_const * (torch.pi - angles)
            else:
                angle_cost = rotation_const * angles

        g2 = g2 + angle_cost

    # rotation_weight: alpha
    # rotation_const: beta
    # g_ratio: lambda

    #
    idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
    idx = idx * neighbor_nodes
    idx = idx.detach()
    g = g2 * idx + g * (1 - idx)

    # update open maps
    open_maps = torch.clamp(open_maps + idx, 0, 1)
    open_maps = open_maps.detach()

    # for backtracking
    idx = idx.reshape(num_samples, -1)
    snm = selected_node_maps.reshape(num_samples, -1)
    new_parents = snm.max(-1, keepdim=True)[1]
    parents = new_parents * idx + parents * (1 - idx)

    return g, h, open_maps, histories, history_probs, parents, is_unsolved