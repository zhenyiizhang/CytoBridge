import random
import torch
import numpy as np
import math
import os
from anndata import AnnData
from torchdiffeq import odeint
from sklearn.decomposition import PCA

from ..tl.models import DynamicalModel

def trace_df_dz(f, z):
    sum_diag = 0.0
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0][:, i]
    return sum_diag

def compute_integral(x, time, V, rho_net, num_samples=100, sigma=0.1, sigmaa=1.0):
    batch_size, dim = x.shape
    noise = torch.randn(batch_size, num_samples, dim, device=x.device)
    y = sigma * noise
    perterbed_x = x.unsqueeze(1) + sigma * noise
    perterbed_x_flat = perterbed_x.reshape(-1, dim)
    time_repeated = time.repeat(num_samples, 1)
    ss = rho_net(time_repeated, perterbed_x_flat)
    rrho = torch.exp(ss * 2 / (sigmaa ** 2))
    rrho = rrho.reshape(batch_size, num_samples, 1)
    y_flat = y.reshape(-1, dim)
    y_flat.requires_grad_(True)

    v = V(y_flat)
    if v.dim() > 1:
        v = v.squeeze(-1)
    grad_y = torch.autograd.grad(v.sum(), y_flat, create_graph=False)[0]
    F = -grad_y
    F = F.view(batch_size, num_samples, dim)
    norm_constant = (2 * math.pi) ** (dim / 2) * (sigma ** dim)
    q = torch.exp(-0.5 * (noise ** 2).sum(dim=-1)) / norm_constant
    contributions = (rrho * F) / q.unsqueeze(-1)
    integral_estimate = contributions.mean(dim=1)
    return integral_estimate

# --------------------------
def set_seed(seed=42):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed)

def sample(data, size):
    N = data.shape[0]
    indices = torch.tensor(random.sample(range(N), size))
    sampled = data[indices]
    return sampled

def check_submodules(model):
    growth_net_exists = hasattr(model, 'growth_net')
    score_net_exists = hasattr(model, 'score_net')
    interaction_net_exists = hasattr(model, 'interaction_net')

    return growth_net_exists, score_net_exists, interaction_net_exists

def load_model_from_adata(adata: AnnData) -> torch.nn.Module:
    if 'all_model' not in adata.uns or 'model_config' not in adata.uns['all_model']:
        raise ValueError("CytoBridge model information not found. Please fit the model first.")

    print("Reconstructing model...")

    import json
    training_config = adata.uns['all_model']['training_config']

    if isinstance(training_config.get('plan'), str):
        training_config['plan'] = json.loads(training_config['plan'])  # Convert JSON string back to list

    model_config = adata.uns['all_model']['model_config']
    dim = adata.obsm['X_latent'].shape[1]
    model = DynamicalModel(dim, model_config)

    state_dict_np = adata.uns['all_model']['model_state_dict']
    state_dict_torch = {k: torch.from_numpy(v) for k, v in state_dict_np.items()}
    model.load_state_dict(state_dict_torch)
    model.eval()

    print("Model loaded successfully.")
    return model

#%%
import torch
from anndata import AnnData
from CytoBridge.tl.models import DynamicalModel
import os

def save_model_to_adata(adata: AnnData, model: DynamicalModel) -> None:
    """
    Save the trained DynamicalModel to the AnnData object

    Parameters:
        adata: AnnData object used to store the model
        model: Trained DynamicalModel instance
    """
    if 'dynamic_model' not in adata.uns:
        adata.uns['dynamic_model'] = {}

    # Save model configuration
    adata.uns['dynamic_model']['model_config'] = model.config

    # Save model weights (convert to numpy array for h5ad compatibility)
    state_dict = model.state_dict()
    state_dict_cpu_numpy = {k: v.cpu().numpy() for k, v in state_dict.items()}
    adata.uns['dynamic_model']['model_state_dict'] = state_dict_cpu_numpy

    print("Model has been successfully saved to adata.uns['dynamic_model']")

def save_model_to_file(model: DynamicalModel, save_path: str) -> None:
    """
    Save the model directly as a pth file (recommended for separate backup)

    Parameters:
        model: Trained DynamicalModel instance
        save_path: Save path (e.g., 'models/trained_model.pth')
    """
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model weights and configuration
    torch.save({
        'model_config': model.config,
        'state_dict': model.state_dict()
    }, save_path)

    print(f"Model has been successfully saved to file: {save_path}")

def load_model_from_file(load_path: str, latent_dim: int) -> DynamicalModel:
    """
    Load model from pth file

    Parameters:
        load_path: Model file path
        latent_dim: Latent space dimension (must match the one used during training)
    Returns:
        DynamicalModel instance with loaded weights
    """
    checkpoint = torch.load(load_path)
    model = DynamicalModel(latent_dim, checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"Model has been loaded from file: {load_path}")
    return model