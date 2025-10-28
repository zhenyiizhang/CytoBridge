# -*- coding: utf-8 -*-

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '/home/sjt/workspace2/CytoBridge_test-main_crufm')))

import scanpy as sc
import CytoBridge

save_path = "/home/sjt/workspace2/CytoBridge_test-main_crufm/results/experiment_simulation/adata.h5ad"      

exp_name      = os.path.basename(os.path.dirname(save_path))      # -> results_unscore
base_fig_dir  = "/home/sjt/workspace2/CytoBridge_test-main_crufm/figures"
exp_fig_dir   = os.path.join(base_fig_dir, exp_name)
os.makedirs(exp_fig_dir, exist_ok=True)

adata  = sc.read_h5ad(save_path)
model  = CytoBridge.utils.load_model_from_adata(adata)
device = "cuda"
CytoBridge.pl.plot_growth(
    adata,
    dim_reduction='umap',
    save_path=os.path.join(exp_fig_dir, "g_values_plot.pdf")
)
if 'velocity_latent' in adata.obsm:
    CytoBridge.pl.plot_velocity_stream(adata, exp_fig_dir, dim_reduction='none', device='cuda')
if 'score_latent' in adata.obsm:
    CytoBridge.pl.plot_score_stream(adata, exp_fig_dir, dim_reduction='none', device='cuda')
if 'score_latent' and 'velocity_latent' in adata.obsm:
    CytoBridge.pl.plot_combined_velocity_stream(adata,exp_fig_dir,dim_reduction='none',device='cuda')




CytoBridge.pl.plot_ode_trajectories(
    adata=adata,
    save_dir=exp_fig_dir,
    n_trajectories=20,
    n_bins=10,

    dim_reduction='none',
    device=device
)

adata = CytoBridge.pl.plot_sde_trajectories(
    adata=adata,
    exp_dir=os.path.join(exp_fig_dir, "sde_results"), 
    device=device,
    sigma=0.05,
    n_time_steps=10,
    sample_traj_num=10,
    sample_cell_tra_consistent = True,
)

for t in [0, 1, 2, 3, 4]:
    CytoBridge.pl.plot_score_and_gradient(
        dynamical_model=model,
        device=device,
        t_value=t,
        x_range=(0, 2.5),
        y_range=(0, 2.5),
        save_path=os.path.join(exp_fig_dir, f"{t}_density.pdf"),
        cmap='rainbow'
    )
