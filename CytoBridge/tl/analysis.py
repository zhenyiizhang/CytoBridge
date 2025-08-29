from CytoBridge.utils import load_model_from_adata
import torch
def compute_velocity(adata, device='cuda'):
    '''
    Calculate velocity based on the trained model 
    '''
    model = load_model_from_adata(adata)
    model.to(device)
    # Calculate velocity and growth on the dataset for downstream analysis
    all_times = torch.tensor(adata.obs['time_point_processed']).unsqueeze(1).float().to(device)
    all_data = torch.tensor(adata.obsm['X_latent']).float().to(device)
    net_input = torch.cat([all_data, all_times], dim = 1)
    velocity = model.velocity_net(net_input)
    adata.obsm['velocity_latent'] = velocity.detach().cpu().numpy()

    return adata