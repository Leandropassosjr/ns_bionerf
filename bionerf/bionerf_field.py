"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Literal, Tuple, Optional, Type

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.vanilla_nerf_field import NeRFField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field

from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP


class BioNeRFField(NeRFField):
    """BioNeRF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_delta_num_layers: Number of layers for Delta MLP.
        base_mlp_c_num_layers: Number of layers for c MLP.
        head_mlp_delta_num_layers: Number of layers for Delta' MLP.
        head_mlp_c_num_layers: Number of layers for c' MLP.
        base_mlp_width: Width of Delta and c MLP layers.
        head_mlp_delta_width: Width of Delta' layers.
        head_mlp_c_width: Width of c' layers.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        base_mlp_delta_num_layers: int = 3,
        base_mlp_c_num_layers: int = 3,
        base_mlp_width: int = 256,
        head_mlp_delta_num_layers: int = 2,
        head_mlp_delta_width: int = 256,
        head_mlp_c_num_layers: int = 1,
        head_mlp_c_width: int = 128,
        field_heads: Optional[Tuple[Type[FieldHead]]] = (RGBFieldHead,)
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding

        self.mlp_base_delta = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_delta_num_layers,
            layer_width=base_mlp_width,
            out_activation=nn.ReLU(),
        )

        self.mlp_head_delta = MLP(
            in_dim=self.position_encoding.get_out_dim() + base_mlp_width,
            num_layers=head_mlp_delta_num_layers,
            layer_width=head_mlp_delta_width,
            out_activation=nn.ReLU(),
        )

        self.mlp_base_c = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_c_num_layers,
            layer_width=base_mlp_width,
            out_activation=nn.ReLU(),
        )

        self.W_gamma = nn.Linear(base_mlp_width*2, base_mlp_width)
        self.W_psi = nn.Linear(base_mlp_width*2, base_mlp_width)
        self.W_mu = nn.Linear(base_mlp_width*2, base_mlp_width)
        self.memory = None

        self.field_output_density = DensityFieldHead(in_dim=self.mlp_head_delta.get_out_dim())

        if field_heads:
            self.mlp_head_c = MLP(
                in_dim=self.direction_encoding.get_out_dim() + base_mlp_width,
                num_layers=head_mlp_c_num_layers,
                layer_width=head_mlp_c_width,
                out_activation=nn.ReLU(),
            )

        self.field_heads = nn.ModuleList([field_head() for field_head in field_heads] if field_heads else [])  # type: ignore
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head_c.get_out_dim())  # type: ignore


    def forward(self, ray_samples: RaySamples) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)

        h_delta = self.mlp_base_delta(encoded_xyz)  
        h_c = self.mlp_base_c(encoded_xyz)     

        # memory
        f_delta = torch.sigmoid(h_delta)
        f_c = torch.sigmoid(h_c)
        h_delta_h_c = torch.cat((h_delta, h_c), dim=-1)

        gamma = torch.tanh(self.W_gamma(h_delta_h_c))
        f_mu = torch.sigmoid(self.W_mu(h_delta_h_c))
        mu = torch.mul(gamma, f_mu)

        if self.memory is None:
            memory = torch.tanh(mu)
            self.memory = memory.detach()
        else:
            if self.memory.shape[0]!=ray_samples.frustums.get_positions().shape[0]:
                f_psi = torch.sigmoid(self.W_psi(h_delta_h_c))

                memory = torch.tanh(mu + torch.mul(f_psi, self.memory[:encoded_xyz.shape[0],:]))
                new_memory = self.memory.clone()
                new_memory[:memory.shape[0],:] = memory
                self.memory = new_memory.detach()

                del new_memory      
                torch.cuda.empty_cache()    


            else:
                f_psi = torch.sigmoid(self.W_psi(h_delta_h_c))
                memory = torch.tanh(mu + torch.mul(f_psi, self.memory))
                self.memory = memory.detach()    

        density = self.get_density(encoded_xyz, torch.mul(f_delta, memory))


        # ------------ ate aqui ------------------

        field_outputs = self.get_outputs(ray_samples, torch.mul(f_c, memory))
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        return field_outputs

    def get_density(self, encoded_xyz: Tensor, density_embedding: Tensor) -> Tensor:
        base_mlp_out = self.mlp_head_delta(torch.cat([encoded_xyz, density_embedding], dim=-1))
        return self.field_output_density(base_mlp_out)
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Tensor) -> Dict[FieldHeadNames, Tensor]:
        outputs = {}
        for field_head in self.field_heads:
            encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
            mlp_out = self.mlp_head_c(torch.cat([encoded_dir, density_embedding], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs