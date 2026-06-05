"""Neural A* search
Author: Ryo Yonetani
Affiliation: OSX
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.actor.neuralastar.differentiable_astar import DifferentiableAstar

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SemanticSegmenterOutput

from src.models.actor.neuralastar.astar import VanillaAstar


class NeuralAstarModule(VanillaAstar):
    def __init__(
        self,
        model: nn.Module,
        g_ratio: float = 0.4,
        Tmax: float = 1.0,
        encoder_input: str = "m+",
        learn_obstacles: bool = False,
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
        
        g_ratio = 0.1

        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=Tmax,
        )
        
        self.encoder_input = encoder_input
        self.encoder = model
        self.learn_obstacles = learn_obstacles
        if self.learn_obstacles:
            print("WARNING: learn_obstacles has been set to True")
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar
        self.encoder = torch.compile(self.encoder)

    def preprocess_inputs(self, images: torch.FloatTensor):
        obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=images.device)
        obstacle_map = torch.sigmoid((images - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        obstacle_map = (~(obstacle_map < 1.51)).float()

        start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=images.device)
        start_map = torch.sigmoid((images - start_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        start_map = (start_map < 1.51).float()

        end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=images.device)
        end_map = torch.sigmoid((images - end_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        end_map = (end_map < 1.51).float()

        return obstacle_map, start_map, end_map

    def encode_cost_maps(self, images: torch.FloatTensor):
        return torch.sigmoid(self.encoder(images).logits.unsqueeze(1))
    
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
        obstacle_map, start_map, end_map = self.preprocess_inputs(images)
        cost_maps = self.encode_cost_maps(images)  # encoder outputs pure logits
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