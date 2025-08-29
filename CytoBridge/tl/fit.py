import scanpy as sc
import torch
from CytoBridge.utils.config import load_config
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.trainer import TrainingPipeline

def fit(adata, config, batch_size = None, device = 'cuda'):
    # Load config
    resolved_config = load_config(config)

    # Set device
    device = torch.device(device)
    
    # Load data
    # TODO: add hold-out here
    time_key = 'time_point_processed'
    time_points = sorted(adata.obs[time_key].unique())
    data_torch = []
    for time_point_i in time_points:
        adata_i = adata[adata.obs[time_key] == time_point_i]
        data_i = adata_i.obsm['X_latent']
        data_i = torch.tensor(data_i).float().to(device)
        data_torch.append(data_i)
    
    if batch_size == None:
        min_n_cells = min([data_i.shape[0] for data_i in data_torch])
        batch_size = min(min_n_cells, 512)
    # Define model
    dim = data_torch[0].shape[1]
    model = DynamicalModel(dim, resolved_config['model'])
    
    # Define trainer
    trainer = TrainingPipeline(model, resolved_config, batch_size, device)

    # Train model
    # TODO: it is possible to save the training curve and checkpoints to a log directory
    model = trainer.train(data_torch, time_points)

    # Evaluate model
    # TODO: if hold-out is used, be careful about the input time points
    trainer.evaluate(data_torch, time_points)

    # Calculate velocity and growth on the dataset for downstream analysis
    all_times = torch.tensor(adata.obs[time_key]).unsqueeze(1).float().to(device)
    all_data = torch.tensor(adata.obsm['X_latent']).float().to(device)
    net_input = torch.cat([all_data, all_times], dim = 1)
    velocity = model.velocity_net(net_input)
    adata.obsm['velocity_latent'] = velocity.detach().cpu().numpy()
    if 'growth' in model.components:
        growth = model.growth_net(net_input)
        adata.obsm['growth_rate'] = growth.detach().cpu().numpy()

    # Store trained results
    # TODO: check whether the model is already trained
    adata.uns['dynamic_model'] = {}

    adata.uns['dynamic_model']['model_config'] = resolved_config['model'] # store the model config

    adata.uns['dynamic_model']['training_config'] = resolved_config['training']

    state_dict = model.state_dict()
    state_dict_cpu_numpy = {k: v.cpu().numpy() for k, v in state_dict.items()}
    adata.uns['dynamic_model']['model_state_dict'] = state_dict_cpu_numpy

    return adata

