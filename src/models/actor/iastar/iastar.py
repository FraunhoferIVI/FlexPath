"""Modified from https://github.com/sair-lab/iAstar/"""

import torch
import torch.nn as nn

from src.models.actor.iastar.dastar import dastar
from src.models.actor.iastar.dastar import *
import src.models.actor.iastar.encoder as encoder
from src.models.actor.iastar.encoder import VGGNet

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SemanticSegmenterOutput


class iastar(nn.Module):
    def __init__(self,
                 g_ratio: float = 0.5,
                 Tmax: float = 1,
                 device:str = "cpu",
                 encoder_input = 2,
                 encoder_arch:str = 'CNN',
                 encoder_depth: int = 4,
                 learn_obstacles:bool = False,
                 const: float = None,
                 is_training:bool = True,
                 store_intermediate_results: bool = False,
                 output_path_list = False,
                 w:float = 1.0):
        super().__init__()
        self.encoder_input = encoder_input
        self.encoder_arch = encoder_arch
        self.encoder_depth = encoder_depth
        self.learn_obstacles = learn_obstacles
        self.const = const
        self.init_encoder()
        self.dastar = dastar(g_ratio=g_ratio,
                              device=device,
                              Tmax=Tmax,
                              w = w,
                              store_intermediate_results=store_intermediate_results,
                              output_path_list=output_path_list,
                              is_training=is_training)
        
        self.device = device
        
        self.path_kernel = torch.tensor([[[1.414,1.,1.414,],
                    [1.,0.,1.],
                    [1.414, 1.,1.414]]],
                device = self.device,
                requires_grad=True).expand(1,1, 3, 3)

    def init_encoder(self):
        print("Using %s as encoder", self.encoder_arch)
        e_arch = getattr(encoder, self.encoder_arch)
        if self.encoder_arch == "CNN":
            self.encoder = e_arch(self.encoder_input,
                                self.encoder_depth,
                                self.const)
        elif "FCN" in self.encoder_arch:
            vgg_model = VGGNet(requires_grad=True)
            self.encoder = e_arch(pretrained_net=vgg_model, n_class=1)
        elif self.encoder_arch=="UNet":
            self.encoder = e_arch(3,1)
        elif self.encoder_arch=="ActorRNT_S_4":
            self.encoder = e_arch()

        # self.encoder = torch.compile(self.encoder)

    def init_obstacles_maps(self, maps):
        obstacle_maps = (
            maps if not self.learn_obstacles else torch.ones_like(maps)
            )
        return obstacle_maps

    def encode(self, maps, start_maps, goal_maps):
        inputs = maps
        if self.encoder_input==3:
            inputs = torch.cat((inputs, start_maps, goal_maps), dim=1)
        elif self.encoder_input==2:
            inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
        cost_maps = self.encoder(inputs)

        return cost_maps
    
    def forward(
        self,
        images: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        starts: Optional[torch.LongTensor] = None,
        ends: Optional[torch.LongTensor] = None,
        loss_fn: nn.Module = None,  # computes loss only if not set to None, uses IL loss all of the time
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        
        # convert to one hot encodings
        obstacle_color = torch.tensor([76 / 255, 76 / 255, 255 / 255], device=images.device)  # BLUE
        obstacle_map = torch.sigmoid((images - obstacle_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        obstacle_map = (~(obstacle_map < 1.51)).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

        start_color = torch.tensor([255 / 255, 76 / 255, 76 / 255], device=images.device)  # BLUE
        start_map = torch.sigmoid((images - start_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        start_map = (start_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

        end_color = torch.tensor([76 / 255, 255 / 255, 76 / 255], device=images.device)  # BLUE
        end_map = torch.sigmoid((images - end_color.view(3, 1, 1)).abs()).sum(1, keepdim=True)
        end_map = (end_map < 1.51).float()  # obstacles will have a value of 3 * sig(0.0) = 1.5, add small margin to be safe against numerical inaccuracies

        planner_outputs = self.forward_original(
            maps=obstacle_map,
            start_maps=start_map,
            goal_maps=end_map
        )

        loss = self.CostofTraj(obstacle_map, planner_outputs)

        if loss_fn is None:
            # Inference mode, no loss calculation
            loss = None
        else:
            # Training mode
            loss = self.CostofTraj(obstacle_map, planner_outputs)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=planner_outputs.paths.squeeze(1),
            hidden_states=planner_outputs.histories.squeeze(1),
            attentions=None,
        )

    def forward_original(
        self,
        maps:torch.tensor,
        start_maps:torch.tensor,
        goal_maps:torch.tensor
    ):
        encoded = self.encode(maps, start_maps, goal_maps)
        return self.dastar(
            encoded,
            start_maps,
            goal_maps,
            self.init_obstacles_maps(maps)
        )

    def CostofTraj(self, maps, outputs,alpha=0.75, beta=0.25):
        paths = outputs.paths
        area_loss = torch.sum(outputs.histories - outputs.paths)/maps.shape[0]
        pad = nn.ReplicationPad2d(padding=(1,1,1,1))
        pad = nn.ZeroPad2d(padding=(1,1,1,1)).to(self.device)
        path_length = F.conv2d(pad(paths).float(), self.path_kernel)
        length_loss = torch.sum(path_length*paths)/(2*maps.shape[0])
        return torch.sqrt(area_loss) + length_loss
