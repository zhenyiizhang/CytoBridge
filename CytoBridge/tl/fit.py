import json
import pathlib
import shutil
from typing import Dict, Any
import scanpy as sc
import torch
import yaml
from CytoBridge.utils.config import load_config
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.trainer import TrainingPipeline


def fit(adata: sc.AnnData,
        config: Dict[str, Any] | str,
        batch_size: int | None = None,
        device: str = 'cuda') -> sc.AnnData:
    """

    TODO: add hold-out split
    TODO: save training curves / checkpoints to log directory
    TODO: support resume from checkpoint
    """
    # ---------- 1. load & resolve config ----------
    resolved_config = load_config(config)  # allow str-path or dict
    device = torch.device(device)

    # ---------- 2. data preparation ----------
    time_key = 'time_point_processed'
    time_points = sorted(adata.obs[time_key].unique())
    data_torch = []
    for t in time_points:
        subset = adata[adata.obs[time_key] == t]
        tens = torch.tensor(subset.obsm['X_latent'], dtype=torch.float32, device=device)
        data_torch.append(tens)

    # ---------- 3. auto batch-size ----------
    if batch_size is None:
        batch_size = min(min(x.shape[0] for x in data_torch), 512)

    # ---------- 4. build & train model ----------
    dim = data_torch[0].shape[1]
    model = DynamicalModel(dim, resolved_config['model'])
    trainer = TrainingPipeline(model, resolved_config, batch_size, device, data=data_torch)
    model = trainer.train(data_torch, time_points)

    # ---------- 5. compute latent outputs ----------
    all_times = torch.tensor(adata.obs[time_key].values, dtype=torch.float32, device=device).unsqueeze(1)
    all_data = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    net_input = torch.cat([all_data, all_times], dim=1)

    velocity = model.velocity_net(net_input)
    adata.obsm['velocity_latent'] = velocity.detach().cpu().numpy()

    if 'growth' in model.components:
        growth = model.growth_net(net_input)
        adata.obsm['growth_rate'] = growth.detach().cpu().numpy()

    # ---------- 6. store model internals ----------
    adata.uns['all_model'] = {
        'model_config': resolved_config['model'],
        'training_config': {
            'defaults': resolved_config['training']['defaults'],
            'plan': json.dumps(resolved_config['training']['plan'])
        },
        'model_state_dict': {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    }

    # ---------- 7. save ----------
    ckpt_dir = pathlib.Path(resolved_config['ckpt_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = ckpt_dir / 'adata.h5ad'
    yaml_path = ckpt_dir / 'config.yaml'

    adata.write_h5ad(h5ad_path)
    with yaml_path.open('w', encoding='utf-8') as yf:
        yaml.dump(resolved_config, yf, default_flow_style=False, allow_unicode=True)

    print(f"Model & data saved -> {ckpt_dir}")
    
    trainer.evaluate(data_torch, time_points)  # TODO handle hold-out

    return adata