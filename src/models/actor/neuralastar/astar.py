"""Neural A* search
Author: Ryo Yonetani
Affiliation: OSX
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.actor.neuralastar.encoder import CNN
from src.models.actor.neuralastar.differentiable_astar import AstarOutput, DifferentiableAstar
from src.models.actor.neuralastar.pq_astar import pq_astar

from src.models.actor.actor_rnt_s_4_transpath_equivalent import ActorRNT_S_4_TP_Equiv

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SemanticSegmenterOutput


class VanillaAstar(nn.Module):
    def __init__(
        self,
        g_ratio: float = 0.5,
        use_differentiable_astar: bool = True,
    ):
        """
        Vanilla A* search

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
            use_differentiable_astar (bool, optional): if the differentiable A* is used instead of standard A*. Defaults to True.

        Examples:
            >>> planner = VanillaAstar()
            >>> outputs = planner(map_designs, start_maps, goal_maps)
            >>> histories = outputs.histories
            >>> paths = outputs.paths

        Note:
            For perform inference on a large map, set use_differentiable_astar = False to peform a faster A* with priority queue
        """

        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=1.0,
        )
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def perform_astar(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        obstacles_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:

        astar = (
            self.astar
            if self.use_differentiable_astar
            else partial(pq_astar, g_ratio=self.g_ratio)
        )

        astar_outputs = astar(
            map_designs,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )

        return astar_outputs

    def forward(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        """
        Perform A* search

        Args:
            map_designs (torch.tensor): map designs (obstacle maps or raw image)
            start_maps (torch.tensor): start maps indicating the start location with one-hot binary map
            goal_maps (torch.tensor): goal maps indicating the goal location with one-hot binary map
            store_intermediate_results (bool, optional): If the intermediate search results are stored in Astar output. Defaults to False.

        Returns:
            AstarOutput: search histories and solution paths, and optionally intermediate search results.
        """

        cost_maps = map_designs
        obstacles_maps = map_designs

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )


class NeuralAstar(VanillaAstar):
    def __init__(
        self,
        g_ratio: float = 0.4,
        Tmax: float = 1.0,
        encoder_input: str = "m+",
        encoder_arch: str = "CNN",
        encoder_depth: int = 4,
        learn_obstacles: bool = False,
        const: float = None,
        use_differentiable_astar: bool = True,
    ):
        """
        Neural A* search

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
            Tmax (float, optional): how much of the map the model explores during training. Set a small value (0.25) when training the model. Defaults to 1.0.
            encoder_input (str, optional): input format. Set "m+" to use the concatenation of map_design and (start_map + goal_map). Set "m" to use map_design only. Defaults to "m+".
            encoder_arch (str, optional): encoder architecture. Defaults to "CNN".
            encoder_depth (int, optional): depth of the encoder. Defaults to 4.
            learn_obstacles (bool, optional): if the obstacle is invisible to the model. Defaults to False.
            const (float, optional): learnable weight to be multiplied for h(v). Defaults to None.
            use_differentiable_astar (bool, optional): if the differentiable A* is used instead of standard A*. Defaults to True.

        Examples:
            >>> planner = NeuralAstar()
            >>> outputs = planner(map_designs, start_maps, goal_maps)
            >>> histories = outputs.histories
            >>> paths = outputs.paths

        Note:
            For perform inference on a large map, set use_differentiable_astar = False to peform a faster A* with priority queue
        """

        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=Tmax,
        )
        self.encoder_input = encoder_input
        if encoder_arch == "checkpoint":
            self.encoder = CNN(input_dim=2, encoder_depth=4)
        else:
            self.encoder = ActorRNT_S_4_TP_Equiv(2, 1, 64, 64, 0.1, keep_x=True)

        self.learn_obstacles = learn_obstacles
        if self.learn_obstacles:
            print("WARNING: learn_obstacles has been set to True")
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def encode(
        self,
        map_designs: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
    ) -> torch.tensor:
        """
        Encode the input problem

        Args:
            map_designs (torch.tensor): map designs (obstacle maps or raw image)
            start_maps (torch.tensor): start maps indicating the start location with one-hot binary map
            goal_maps (torch.tensor): goal maps indicating the goal location with one-hot binary map

        Returns:
            torch.tensor: predicted cost maps
        """
        inputs = map_designs
        if "+" in self.encoder_input:
            if map_designs.shape[-1] == start_maps.shape[-1]:
                inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
            else:
                upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
                inputs = torch.cat((inputs, upsampler(start_maps + goal_maps)), dim=1)

        cost_maps = self.encoder(inputs)

        return cost_maps
    
    def forward(self,
        images: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        starts: Optional[torch.LongTensor] = None,
        ends: Optional[torch.LongTensor] = None,
        loss_fn: nn.Module = None,  # will always use 
        store_intermediate_results: bool = False,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        """
        Perform neural A* search

        Args:
            map_designs (torch.tensor): map designs (obstacle maps or raw image)
            start_maps (torch.tensor): start maps indicating the start location with one-hot binary map
            goal_maps (torch.tensor): goal maps indicating the goal location with one-hot binary map
            store_intermediate_results (bool, optional): If the intermediate search results are stored in Astar output. Defaults to False.

        Returns:
            AstarOutput: search histories and solution paths, and optionally intermediate search results.
        """

        obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=images.device)  # BLUE
        obstacle_map = torch.sigmoid((images - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        obstacle_map = (~(obstacle_map < 1.51)).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

        start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=images.device)  # BLUE
        start_map = torch.sigmoid((images - start_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        start_map = (start_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

        end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=images.device)  # BLUE
        end_map = torch.sigmoid((images - end_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        end_map = (end_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

        cost_maps = F.sigmoid(self.encode(obstacle_map, start_map, end_map).logits.unsqueeze(1))  # encoder outputs pure logits
        # cost_maps = F.sigmoid(self.encode(obstacle_map, start_map, end_map))  # FOR CHECKPOINT
        obstacles_maps = (
            obstacle_map if not self.learn_obstacles else torch.ones_like(start_map)
        )

        astar_outputs = self.perform_astar(
            cost_maps,
            start_map,
            end_map,
            obstacles_maps,
            store_intermediate_results,
        )

        if loss_fn is not None: 
            # training API, 
            loss = F.l1_loss(astar_outputs.histories.squeeze(1), labels.float(), reduction="mean")
        else:
            loss = None
        
        return SemanticSegmenterOutput(
                loss=loss,
                logits=astar_outputs.paths.squeeze(1),
                hidden_states=astar_outputs.histories.squeeze(1),
                attentions=None,
            )