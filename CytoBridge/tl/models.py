import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Dict, Any, Optional
from CytoBridge.tl.interaction import ExpNormalSmearing, CosineCutoff, cal_interaction
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


class InteractionModel(nn.Module):
    def __init__(self, x_dim,n_layers=2 , hidden_dim=400, activation='leaky_relu', num_rbf=16, cutoff=1, dim_reduce = False, residual = False):
        if activation not in ACTIVATION_FN:
            raise ValueError(f"Activation '{activation}' not recognized.")

        self.n_layers = n_layers
        self.residual = residual
        act_fn = ACTIVATION_FN[activation]
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.rbf_expansion = ExpNormalSmearing(cutoff=cutoff, num_rbf = self.num_rbf, trainable= True)

        if n_layers == 0:
            self.input_layer = nn.Linear(self.num_rbf, 1)
            self.hidden_layers = nn.ModuleList([])
            self.output_layer = nn.Identity()
        else:
            self.input_layer = nn.Sequential(
                nn.Linear(self.num_rbf, hidden_dim),
                act_fn()
            )
            self.hidden_layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    act_fn()
                )
                for _ in range(n_layers - 1)
            )
            self.output_layer = nn.Linear(hidden_dim, 1)

        self.cutoff = cutoff
        self.eps = 1e-6
        self.dim_reduce = dim_reduce
        if self.dim_reduce:
            self.pca = nn.Linear(x_dim, 10, bias=False)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x_t):
        if self.cutoff == 0:
            return 0 * x_t.sum()
        if self.dim_reduce:
            x_t = self.pca(x_t)
        dis = self.compute_distance(x_t)
        dis_exp = self.rbf_expansion(dis[dis != 0])

        if self.n_layers == 0:
            return self.input_layer(dis_exp)
        x = self.input_layer(dis_exp)
        for layer in self.hidden_layers:
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)
        potential = self.output_layer(x)
        return potential

    def compute_distance(self, x):
        return torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.eps)


class DynamicalModel(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            config: Dict[str, Any],
            use_growth_in_ode_inter = True
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

            elif comp_name == 'score':

                network = HyperNetwork(
                    input_dim=self.net_input_dim,
                    output_dim=1,
                    **comp_config
                )

            elif comp_name == 'interaction':
                self.use_growth_in_ode_inter = use_growth_in_ode_inter 
                network = InteractionModel(self.latent_dim, **comp_config)
            else:
                raise ValueError(f"Unknown dynamical component: '{comp_name}'")

            self.add_module(f"{comp_name}_net", network)

    def forward(self, t: torch.Tensor, x: torch.Tensor, lnw: torch.Tensor,except_interaction:bool = True) -> Dict[str, torch.Tensor]:
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

        # Handle score component
        if 'score' in self.components:
            x = x.requires_grad_(True)
            net_input = torch.cat([x, t_expanded], dim=1)
            out_score = self.score_net(net_input)
            outputs['score'] = out_score
            gradient = torch.autograd.grad(
                outputs=out_score,
                inputs=x,
                grad_outputs=torch.ones_like(out_score),
                create_graph=True
            )[0]
            outputs['score_gradient'] = gradient

        # Handle interaction component
        if 'interaction' in self.components and except_interaction :
            #print("self.use_growth_in_ode_inter", self.use_growth_in_ode_inter)
            outputs['interaction'] = cal_interaction(x, lnw, self.interaction_net, cutoff = self.interaction_net.cutoff, use_mass = self.use_growth_in_ode_inter).float()

        return outputs

    def compute_score(self, t, x):
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        t_expanded = t.expand(x.size(0), 1)
        x = x.requires_grad_(True)
        net_input = torch.cat([x, t_expanded], dim=1)
        out_score = self.score_net(net_input)
        gradient = torch.autograd.grad(
                outputs=out_score,
                inputs=x,
                grad_outputs=torch.ones_like(out_score),
                create_graph=True
            )[0]
        return out_score, gradient
