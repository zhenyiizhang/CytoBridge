import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

ACTIVATION_FN = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}


class HyperNetwork(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 400,
            n_layers: int = 2,
            activation: str = 'leaky_relu',
            residual: bool = False
    ):
        super().__init__()

        if activation not in ACTIVATION_FN:
            raise ValueError(f"Activation '{activation}' not recognized.")

        self.n_layers = n_layers
        self.residual = residual
        act_fn = ACTIVATION_FN[activation]

        if self.n_layers == 0:
            self.input_layer = nn.Linear(input_dim, output_dim)
            self.hidden_layers = nn.ModuleList([])
            self.output_layer = nn.Identity()
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_fn()
            )

            self.hidden_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        act_fn()
                    )
                    for _ in range(n_layers - 1)
                ]
            )
            self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_layers == 0:
            return self.input_layer(x)

        x = self.input_layer(x)

        for layer in self.hidden_layers:
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)

        x = self.output_layer(x)
        return x




class DynamicalModel(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            config: Dict[str, Any],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.components = config['components']
        self.net_input_dim = self.latent_dim + 1  # 状态+时间（x + t）

        for comp_name in self.components:
            net_name = f'{comp_name}_net'
            if net_name not in self.config:
                raise ValueError(f"Configuration for component '{comp_name}' not found.")

            comp_config = self.config[net_name].copy()  

            if comp_name == 'velocity':
                network = HyperNetwork(
                    input_dim=self.net_input_dim,
                    output_dim=self.latent_dim,
                    **comp_config
                )

            elif comp_name == 'growth':
                network = HyperNetwork(
                    input_dim=self.net_input_dim,
                    output_dim=1, **comp_config
                )


            # TODO: add interaction results
            else:
                raise ValueError(f"Unknown dynamical component: '{comp_name}'")

            self.add_module(f"{comp_name}_net", network)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Ensure t is a 2D tensor [batch_size, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)  # 从[batch_size] -> [batch_size, 1]
        t_expanded = t.expand(x.size(0), 1)  # Ensure time dimension matches x

        outputs = {}
        # Handle velocity component
        if 'velocity' in self.components:
            net_input = torch.cat([x, t_expanded], dim=1)
            outputs['velocity'] = self.velocity_net(net_input)

        # Handle growth component
        if 'growth' in self.components:
            net_input = torch.cat([x, t_expanded], dim=1)
            outputs['growth'] = self.growth_net(net_input)


        return outputs