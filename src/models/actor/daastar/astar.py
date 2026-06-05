# -------------------------------------------------------------------------------------------------------------
# File: astar.py
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

import torch
import os
import sys
import torch.nn as nn

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../..")

from functools import partial
from copy import deepcopy
from src.models.actor.daastar.differentiable_astar import DifferentiableAstar
from src.models.actor.daastar.pq_astar import pq_astar

from src.models.actor.daastar.encoder import ActorRNT_S_4
from src.models.actor.daastar.modules import Encoder, Decoder, PosEmbeds, SpatialTransformer



# =============================================================================================================

class VanillaAstar(nn.Module):
    def __init__(
        self,
        config,
        use_differentiable_astar=True,
        disable_heuristic=False
    ):
        super().__init__()
        config_modified = deepcopy(config)
        config_modified.Tmax = 1.0
        config_modified.expand_mode = 'max'

        
        self.astar = DifferentiableAstar(
            config_modified,
            disable_heuristic=disable_heuristic
        )

        # self.astar = NeuralDifferentiableAstar(
        #     g_ratio=0.13,
        # )  # only for time measurements

        self.config = config_modified
        self.use_differentiable_astar = use_differentiable_astar
        self.learn_obstacles = config.dataset_name in ['sdd_inter', 'sdd_intra', 'warcraft', 'pkmn']
        if self.learn_obstacles: print("WARNING: learn_obstacles has been set to True")

        if not use_differentiable_astar and config_modified.enable_angle:
            assert False, '!!!Error: pqastar does not support angle constraint.'

    def encode(
        self,
        map_designs,
        start_maps,
        goal_maps
    ):
        return None

    def encode_transpath(
        self,
        map_designs,
        start_maps,
        goal_maps,
    ):
        return None

    def data_preprocessing(
        self,
        map_designs,
        start_maps,
        goal_maps,
        prob_maps=None
    ):
        # NOTE: for warcraft 12*12: input image is 3*(12*8)*(12*8), each grid has 8 pixels,
        # so map_designs: (3, 96, 96); cost_maps: (1, 12, 12),
        # where 12 is the number of grids in one dimenstion, either h or w.
        # For the binary input images, the shapes of map_designs and cost_maps are the same.
        dataset_name = self.config.dataset_name
        pred_maps = None
        obstacles_maps = None

        if prob_maps is None:
            # print("Encoding...")
            pred_prob_maps = self.encode_transpath(map_designs, start_maps, goal_maps)
            encoder_const = self.config.encoder.const
        else:
            pred_prob_maps = prob_maps
            encoder_const = 10.0

        # print("Prob1", pred_prob_maps.min(), pred_prob_maps.max())

        # The output is a PPM since it wil be MSE with GT PPM
        pred_maps = pred_prob_maps

        cost_maps = (1 - pred_prob_maps) if pred_prob_maps is not None else None
        cost_maps = encoder_const * cost_maps if cost_maps is not None else None
        obstacles_maps = 1 - map_designs
        
        # print("Prob2", cost_maps.min(), cost_maps.max())

        return cost_maps, obstacles_maps, pred_maps

    def perform_astar(
        self,
        map_designs,
        start_maps,
        goal_maps,
        obstacles_maps,
        store_intermediate_results=False,
        store_hist_coordinates=False,
        disable_heuristic=False,
        disable_compute_path_angle=False
    ):
        astar = (
            self.astar
            if self.use_differentiable_astar
            else partial(pq_astar, g_ratio=self.config.g_ratio)
        )

        astar_outputs = astar(
            map_designs,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
            store_hist_coordinates,
            disable_heuristic,
            disable_compute_path_angle
        )

        return astar_outputs

    def forward(
        self,
        map_designs,
        start_maps,
        goal_maps,
        store_intermediate_results=False,
        store_hist_coordinates=False,
        disable_compute_path_angle=False
    ):
        cost_maps, obstacles_maps, _ = self.data_preprocessing(
            map_designs,
            start_maps,
            goal_maps
        )

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
            store_hist_coordinates,
            disable_compute_path_angle=disable_compute_path_angle
        ), None

# =============================================================================================================

class NeuralAstar(VanillaAstar):
    def __init__(
        self,
        config,
        disable_heuristic=False,
        use_differentiable_astar=True
    ):
        super().__init__(config)

        # disabled in DAA* config
        # if config.transpath.enable_diag_astar:
            # disabled in DAA* config
            # # Only for TransPath as it is in their paper
            # assert self.config.transpath.mode == 'f', \
            #     f"Only f mode is supported, unknown mode: {self.config.transpath.mode}."

            # self.astar = DifferentiableDiagAstar(
            #     config.g_ratio,
            #     config.Tmax,
            #     f_w=100,
            #     mode=self.config.transpath.mode
            # )
        # else:
        # This is where we impose angular constraints
        self.astar = DifferentiableAstar(
            config,
            disable_heuristic=disable_heuristic
        )

        # MOD
        self.encoder = ActorRNT_S_4()  # use unified model for better comparison
        # self.construct_transformer_encoder()  # for checkpoint

        self.config = config
        self.use_differentiable_astar = use_differentiable_astar

    def load_transpath_pretrained_model(
        self,
        checkpoint_path
    ):
        state_dict = torch.load(checkpoint_path)["state_dict"]
        module_keys = ['encoder.', 'pos.', 'transformer.', 'decoder_pos.', 'decoder.', 'astar.']
        modules = [self.encoder, self.pos, self.transformer, self.decoder_pos, self.decoder, self.astar]

        for module_key, module in zip(module_keys, modules):
            module_state_dict = {}

            for key in state_dict.keys():
                if key.startswith("planner." + module_key):
                    module_state_dict.update({key.replace(module_key, ''): state_dict[key]})

            # strip 'planner.' prefix
            stripped_module_state_dict = {
                k.removeprefix("planner."): v
                for k, v in module_state_dict.items()
            }
            if "astar" in module_key:
                print(module_key, stripped_module_state_dict.keys())

            module.load_state_dict(stripped_module_state_dict, strict=False)

    def construct_transformer_encoder(self):
        hidden_channels = 64
        downsample_steps = 3
        attn_dropout = 0.15
        cnn_dropout = 0.15
        attn_heads = 4
        mode = self.config.transpath.mode

        in_channels = 2
        out_channels = 1
        attn_blocks = 4
        resolution = (64, 64)
        self.k = 64 * 64 if mode == 'h' else 1

        heads_dim = hidden_channels // attn_heads

        self.encoder = Encoder(
            in_channels,
            hidden_channels,
            downsample_steps,
            cnn_dropout
        )

        self.pos = PosEmbeds(
            hidden_channels,
            (resolution[0] // 2 ** downsample_steps,
             resolution[1] // 2 ** downsample_steps)
        )

        self.transformer = SpatialTransformer(
            hidden_channels,
            attn_heads,
            heads_dim,
            attn_blocks,
            attn_dropout
        )

        self.decoder_pos = PosEmbeds(
            hidden_channels,
            (resolution[0] // 2 ** downsample_steps,
             resolution[1] // 2 ** downsample_steps)
        )

        self.decoder = Decoder(
            hidden_channels,
            out_channels,
            downsample_steps,
            cnn_dropout
        )

    # def encode_transpath(
    #     self,
    #     map_designs,
    #     start_maps,
    #     goal_maps
    # ):
    #     map_designs = 1 - map_designs  # invert
    #     if self.config.dataset_name == 'aug_tmpd':
    #         if self.config.transpath.mode in ('f', 'nastar'):
    #             inputs = torch.cat([map_designs, start_maps + goal_maps], dim=1)
    #         else:
    #             inputs = torch.cat([map_designs, goal_maps], dim=1)
    #     else:
    #         sg = torch.cat((start_maps, goal_maps), dim=1)
    #         inputs = torch.cat([map_designs, sg], dim=1)

    #     x = self.encoder(inputs)
    #     x = self.pos(x)
    #     x = self.transformer(x)
    #     x = self.decoder_pos(x)
    #     x = self.decoder(x)  # decoder has tanh at last
    #     x = (x + 1) / 2  # self.k would always be 1 for this setting

    #     return x

    def encode_transpath(
        self,
        map_designs,
        start_maps,
        goal_maps
    ):
        if self.config.dataset_name == 'aug_tmpd':
            if self.config.transpath.mode in ('f', 'nastar'):
                inputs = torch.cat([map_designs, start_maps + goal_maps], dim=1)
        else:
            sg = torch.cat((start_maps, goal_maps), dim=1)
            inputs = torch.cat([map_designs, sg], dim=1)

        # x = self.encoder(inputs)
        # x = self.pos(x)
        # x = self.transformer(x)
        # x = self.decoder_pos(x)
        # x = self.decoder(x)  # decoder has tanh at last
        # x = (x + 1) * self.k / 2

        ### use our backbone instead
        x = self.encoder(inputs)  # logits
        x = torch.nn.functional.tanh(x)  # squash

        x = (x + 1) / 2  # self.k would always be 1 for this setting

        return x

    def encode(
        self,
        map_designs,
        start_maps,
        goal_maps
    ):
        inputs = map_designs

        if "+" in self.config.encoder.input:
            if map_designs.shape[-1] == start_maps.shape[-1]:
                print("Concat input")
                inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
            else:
                print("Upsampling input")
                upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
                inputs = torch.cat((inputs, upsampler(start_maps + goal_maps)), dim=1)

        cost_maps = self.encoder(inputs)

        return cost_maps

    def forward(
        self,
        map_designs,
        start_maps,
        goal_maps,
        prob_maps=None,
        store_intermediate_results=False,
        store_hist_coordinates=False,
        disable_heuristic=False,
        disable_compute_path_angle=False
    ):
        cost_maps, obstacles_maps, pred_maps = self.data_preprocessing(
            map_designs,
            start_maps,
            goal_maps,
            prob_maps
        )

        astar_outputs = self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
            store_hist_coordinates,
            disable_heuristic,
            disable_compute_path_angle=disable_compute_path_angle
        )

        return astar_outputs, pred_maps