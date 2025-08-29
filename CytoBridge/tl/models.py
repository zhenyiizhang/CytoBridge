import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

# A dictionary to map string names to activation functions
ACTIVATION_FN = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
}

class HyperNetwork(nn.Module):
    """A flexible residual Multi-Layer Perceptron (MLP) builder.

    This network is constructed with an input layer, a series of optional
    residual hidden layers, and a final output layer. The residual connections
    are applied to each hidden layer.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 400,
        n_layers: int = 2,
        activation: str = 'silu',
        residual: bool = False
    ):
        """
        Initializes the HyperNetwork.

        Parameters
        ----------
        input_dim
            Dimension of the input features.
        output_dim
            Dimension of the output features.
        hidden_dim
            Number of neurons in the hidden layers. This must be equal to input_dim
            if n_layers > 0 and residual connections are used.
        n_layers
            Number of hidden layers.
        activation
            Activation function to use. Options: 'relu', 'leaky_relu', 'silu', 'tanh'.
        residual
            If True, applies a residual connection to each hidden layer.
            For this to work, hidden_dim must be equal to the layer's input dimension.
        """
        super().__init__()
        
        if activation not in ACTIVATION_FN:
            raise ValueError(f"Activation '{activation}' not recognized.")
        
        self.n_layers = n_layers
        self.residual = residual
        act_fn = ACTIVATION_FN[activation]

        if self.n_layers == 0:
            # If no hidden layers, it's a direct linear mapping
            self.input_layer = nn.Linear(input_dim, output_dim)
            self.hidden_layers = nn.ModuleList([])
            self.output_layer = nn.Identity() # No-op
        else:
            # Input layer maps from input_dim to hidden_dim
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_fn()
            )

            # Hidden layers are a ModuleList of residual blocks
            self.hidden_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        act_fn()
                    )
                    for _ in range(n_layers -1) # n_layers includes the first layer's mapping
                ]
            )
            # Output layer maps from hidden_dim to output_dim
            self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        if self.n_layers == 0:
            # Handle the no-hidden-layer case
            return self.input_layer(x)

        # Pass through the input layer first
        x = self.input_layer(x)

        # Pass through each hidden layer with optional residual connection
        for layer in self.hidden_layers:
            if self.residual:
                # Add input of the block to its output
                x = x + layer(x)
            else:
                x = layer(x)
        
        # Pass through the final output layer
        x = self.output_layer(x)
        return x



class DynamicalModel(nn.Module):
    """
    A container for the complete dynamical system model.

    This model dynamically constructs and holds the neural networks required
    for a specific OT-based method (e.g., TIGON, DeepRUOT, CytoBridge) based on
    a given configuration.
    """
    def __init__(
        self,
        latent_dim: int,
        config: Dict[str, Any],
    ):
        """
        Initializes the DynamicalModel.

        Parameters
        ----------
        latent_dim
            The dimensionality of the latent space where the dynamics occur.
        components
            A list of strings specifying which dynamical components to include.
            Example: ['velocity', 'growth', 'score', 'interaction']
        config
            A dictionary containing the hyperparameters for each component's network.
            The keys should match the names in `components`.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.components = config['components']

        # The input to most networks is the latent state + time (a scalar)
        # Therefore, the input dimension is latent_dim + 1
        net_input_dim = self.latent_dim + 1

        # Dynamically build the required neural network components
        for comp_name in self.components:
            net_name = f'{comp_name}_net'
            if net_name not in self.config:
                raise ValueError(f"Configuration for component '{comp_name}' not found.")

            comp_config = self.config[net_name]
            
            if comp_name == 'velocity':
                # Velocity network: maps (state, t) -> state_delta
                # TODO: Another way to calculate the velocity is by calculating the gradients of a scaler output (Var-RUOT)
                network = HyperNetwork(input_dim=net_input_dim, output_dim=self.latent_dim, **comp_config)
            
            elif comp_name == 'growth':
                # Growth network: maps (state, t) -> growth_rate (scalar)
                network = HyperNetwork(input_dim=net_input_dim, output_dim=1, **comp_config)
            
            # TODO: add score network
            # TODO: add interaction network    
            else:
                raise ValueError(f"Unknown dynamical component: '{comp_name}'")
            
            # Use setattr to dynamically add the network as a module
            # e.g., self.velocity_net = network
            self.add_module(f"{comp_name}_net", network)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the entire dynamical system at a given time.

        This signature is compatible with torchdiffeq.odeint.

        Parameters
        ----------
        t
            A scalar tensor representing the current time.
        x
            A tensor of shape (batch_size, latent_dim) representing the cell states.

        Returns
        -------
        A dictionary containing the outputs of all active components.
        Example: {'velocity': tensor, 'growth': tensor}
        """
        # Prepare the input for the networks by concatenating state and time
        # We need to expand 't' to match the batch size of 'x'
        t_expanded = t.expand(x.size(0), 1)
        net_input = torch.cat([x, t_expanded], dim=1)

        outputs = {}
        # Get outputs from all registered components
        if 'velocity' in self.components:
            outputs['velocity'] = self.velocity_net(net_input)
        if 'growth' in self.components:
            outputs['growth'] = self.growth_net(net_input)

        # TODO: add score results and interaction results

        return outputs