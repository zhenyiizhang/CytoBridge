import random
import torch
from anndata import AnnData
from CytoBridge.tl.models import DynamicalModel

def sample(data, size):
    N = data.shape[0]
    indices = torch.tensor(random.sample(range(N), size))
    sampled = data[indices]
    return sampled

def load_model_from_adata(adata: AnnData) -> torch.nn.Module:
    if 'dynamic_model' not in adata.uns or 'model_config' not in adata.uns['dynamic_model']:
        raise ValueError("Cannot find CytoBridge model informatio, please fit the model first")

    print("Reconstructing model...")
    
    # 1. Get model architecture configuration
    model_config = adata.uns['dynamic_model']['model_config']

    dim = adata.obsm['X_latent'].shape[1]
    
    # 2. Reconstruct model skeleton using the configuration
    model = DynamicalModel(dim, model_config)
    
    # 3. Get saved weights
    state_dict_np = adata.uns['dynamic_model']['model_state_dict']
    
    # 4. Convert NumPy arrays back to PyTorch tensors
    state_dict_torch = {k: torch.from_numpy(v) for k, v in state_dict_np.items()}
    
    # 5. Load weights into the model
    model.load_state_dict(state_dict_torch)
    
    model.eval()
    
    print("Model loaded successfully.")
    return model