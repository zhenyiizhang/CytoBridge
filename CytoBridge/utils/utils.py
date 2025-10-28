import random
import torch
import numpy as np
from anndata import AnnData
from torchdiffeq import odeint
from sklearn.decomposition import PCA

from ..tl.models import DynamicalModel


def sample(data, size):
    """
    Randomly samples a subset of data points from the input data.
    
    Args:
        data: Input data array/tensor from which to sample, shape (num_samples, ...).
        size: Number of samples to extract.
    
    Returns:
        Sampled array/tensor containing the sampled data points, shape (size, ...).
    """
    N = data.shape[0]
    indices = torch.tensor(random.sample(range(N), size))
    sampled = data[indices]
    return sampled


def load_model_from_adata(adata: AnnData) -> torch.nn.Module:
    """
    Loads a pre-trained DynamicalModel from an AnnData object, where model information is stored in adata.uns.
    
    Args:
        adata: AnnData object containing model information in its 'uns' attribute.
    
    Raises:
        ValueError: If no model information is found in adata.uns (i.e., 'all_model' or its subkeys are missing).
    
    Returns:
        A trained DynamicalModel instance with weights loaded.
    """
    if 'all_model' not in adata.uns or 'model_config' not in adata.uns['all_model']:
        raise ValueError("Cannot find CytoBridge model information, please fit the model first")

    print("Reconstructing model...")

    # 1. Load training configuration (handle JSON string if necessary)
    import json
    training_config = adata.uns['all_model']['training_config']

    # Convert plan from JSON string back to list if needed
    if isinstance(training_config.get('plan'), str):
        training_config['plan'] = json.loads(training_config['plan'])

    # 2. Reconstruct the model using stored configuration
    model_config = adata.uns['all_model']['model_config']
    dim = adata.obsm['X_latent'].shape[1]  # Latent space dimension from stored data
    model = DynamicalModel(dim, model_config)

    # 3. Load model weights (convert numpy arrays back to PyTorch tensors)
    state_dict_np = adata.uns['all_model']['model_state_dict']
    state_dict_torch = {k: torch.from_numpy(v) for k, v in state_dict_np.items()}
    model.load_state_dict(state_dict_torch)
    model.eval()  # Set to evaluation mode

    print("Model loaded successfully.")
    return model


#%%
import torch
from anndata import AnnData
from CytoBridge.tl.models import DynamicalModel
import os


def save_model_to_adata(adata: AnnData, model: DynamicalModel) -> None:
    """
    Saves a trained DynamicalModel into the 'uns' attribute of an AnnData object for easy storage/transport.
    
    Args:
        adata: AnnData object where the model will be stored.
        model: Trained DynamicalModel instance to save.
    """
    if 'dynamic_model' not in adata.uns:
        adata.uns['dynamic_model'] = {}

    # Save model configuration
    adata.uns['dynamic_model']['model_config'] = model.config

    # Save model weights (convert to numpy arrays for compatibility with h5ad format)
    state_dict = model.state_dict()
    state_dict_cpu_numpy = {k: v.cpu().numpy() for k, v in state_dict.items()}
    adata.uns['dynamic_model']['model_state_dict'] = state_dict_cpu_numpy

    print("Model successfully saved to adata.uns['dynamic_model']")


def save_model_to_file(model: DynamicalModel, save_path: str) -> None:
    """
    Saves a trained DynamicalModel directly to a .pth file (recommended for standalone backups).
    
    Args:
        model: Trained DynamicalModel instance to save.
        save_path: Path to save the model (e.g., 'models/trained_model.pth').
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model configuration and weights
    torch.save({
        'model_config': model.config,
        'state_dict': model.state_dict()
    }, save_path)

    print(f"Model successfully saved to file: {save_path}")


def load_model_from_file(load_path: str, latent_dim: int) -> DynamicalModel:
    """
    Loads a trained DynamicalModel from a .pth file.
    
    Args:
        load_path: Path to the .pth file containing the saved model.
        latent_dim: Dimension of the latent space (must match the dimension used during training).
    
    Returns:
        A trained DynamicalModel instance with weights loaded, set to evaluation mode.
    """
    checkpoint = torch.load(load_path)
    model = DynamicalModel(latent_dim, checkpoint['model_config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from file: {load_path}")
    return model