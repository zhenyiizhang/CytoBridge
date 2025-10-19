import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

#  TODO: add dim_reduction

def plot_growth(adata, dim_reduction = 'umap', save_path = "figures/"):
    if 'growth_rate' not in adata.obsm.keys():
        raise ValueError("Growth rate not found in adata.obsm. Please check if the growth term is set or the model is trained.")
    
    # Get the data for the plot
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm.keys():
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        plot_data = adata.obsm['X_umap']
    elif dim_reduction == 'PCA':
        plot_data = adata.obsm['X_pca'][:,:2]
    elif dim_reduction == 'none' or dim_reduction == None:
        plot_data = adata.obsm['X_latent'][:,:2]
    else:
        raise ValueError(f"Invalid dim_reduction: {dim_reduction}")


    g_values = adata.obsm['growth_rate']
    g_min = g_values.min()
    g_max = g_values.max()

    if g_max <= g_min:
        vmin, vmax = -1, 1
    elif g_max < 0:
        vmin, vmax = g_max, 0
    else:
        vmin = max(0, g_min)
        vmax = np.percentile(g_values, 95)

    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=True)
    colors = plt.cm.RdYlBu_r(norm(g_values))

    # Plot the growth values
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(plot_data[:,0], plot_data[:,1], c=colors, alpha=0.8, marker='o', s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array(g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Predicted Growth Rate')

    # Format colorbar ticks
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{(x):.2f}'))
    save_path= os.path.join(save_path, "g_values_plot.pdf")
    # Save figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

    #plt.show()


#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from CytoBridge.utils.utils import load_model_from_adata
from CytoBridge.tl.analysis import generate_ode_trajectories
from scipy.interpolate import make_interp_spline
import pickle
from sklearn.decomposition import PCA
import umap
# %% English version: plot_ode_trajectories + plot_ode
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from CytoBridge.utils.utils import load_model_from_adata
from CytoBridge.tl.analysis import generate_ode_trajectories
from scipy.interpolate import CubicSpline
import pickle
from sklearn.decomposition import PCA
import umap


def plot_ode_trajectories(
    adata,
    save_dir='figures',
    save_base_name='ode_trajectories.png',
    n_trajectories=50,
    n_bins=100,
    dim_reduction='umap',
    device="cuda",
    split_true=False,
    plot_trajectories=50
):
    """Generate ODE trajectories, reduce to 2-D, and save plots."""
    model = load_model_from_adata(adata)
    exp_dir = os.path.join(save_dir, "ode_results")
    os.makedirs(exp_dir, exist_ok=True)

    _, traj_array = generate_ode_trajectories(
        model, adata,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
        exp_dir=exp_dir,
        split_true=split_true
    )

    # No filtering – simply take the first plot_trajectories traces
    num_avail = traj_array.shape[1]
    take = min(plot_trajectories, num_avail)
    selected_traj_array = traj_array[:, :take, :]
    print(f"[Trajectory selection] Using the first {take} trajectories.")

    # Prepare raw data
    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())
    X_raw = adata.obsm['X_latent']
    D = selected_traj_array.shape[-1]

    X_raw_train = X_raw[~np.isnan(X_raw).any(axis=1)]
    if len(X_raw_train) == 0:
        raise ValueError("All raw data contain NaN – cannot train reducer.")

    # Dimensionality reduction
    traj_reshaped = selected_traj_array.reshape(-1, D)
    traj_valid_mask = ~np.isnan(traj_reshaped).any(axis=1)
    traj_reshaped_valid = traj_reshaped[traj_valid_mask]

    if D == 2:
        traj_2d = selected_traj_array
        X_2d = X_raw
    else:
        reducer_pkl = os.path.join(exp_dir, "dim_reducer.pkl")
        if dim_reduction == 'none':
            traj_2d = selected_traj_array[..., :2]
            X_2d = X_raw[..., :2]
        elif dim_reduction in ('pca', 'umap'):
            reducer = None
            if os.path.isfile(reducer_pkl):
                try:
                    with open(reducer_pkl, 'rb') as f:
                        reducer = pickle.load(f)
                except Exception:
                    pass
            if reducer is None or not (hasattr(reducer, 'n_components') and reducer.n_components == 2):
                print(f"[Dim reduction] Training {dim_reduction.upper()} model.")
                if dim_reduction == 'pca':
                    reducer = PCA(n_components=2, random_state=0)
                else:
                    reducer = umap.UMAP(n_components=2, random_state=0)
                reducer.fit(X_raw_train)
                with open(reducer_pkl, 'wb') as f:
                    pickle.dump(reducer, f)

            X_2d = np.full((X_raw.shape[0], 2), np.nan)
            valid_X_mask = ~np.isnan(X_raw).any(axis=1)
            X_2d[valid_X_mask] = reducer.transform(X_raw[valid_X_mask])

            traj_2d = np.full((traj_reshaped.shape[0], 2), np.nan)
            if traj_reshaped_valid.size > 0:
                traj_2d[traj_valid_mask] = reducer.transform(traj_reshaped_valid)
            traj_2d = traj_2d.reshape(selected_traj_array.shape[0], take, 2)
        else:
            raise ValueError("dim_reduction must be 'none', 'pca' or 'umap'.")

    X = [X_2d[adata.obs[samples_key] == t] for t in unique_times]

    # Generate key points: n_time = unique_times.shape[0]
    n_time = len(unique_times)
    time_idx = np.round(np.linspace(0, n_bins - 1, n_time)).astype(int)
    point_array = np.full((n_time, take, 2), np.nan)
    for t_idx, t in enumerate(time_idx):
        for j in range(take):
            traj = traj_2d[:, j, :]
            if t < traj.shape[0] and not np.isnan(traj[t]).any():
                point_array[t_idx, j] = traj[t]

    # Save
    np.save(os.path.join(exp_dir, "ode_traj_2d.npy"), traj_2d)
    np.save(os.path.join(exp_dir, "ode_selected_plot_traj_2d.npy"), traj_2d)
    np.save(os.path.join(exp_dir, "ode_point_2d.npy"), point_array)

    # Plot
    save_path = os.path.join(exp_dir, save_base_name)
    plot_ode(X=X,
             point_array=point_array,
             traj_array=traj_2d,
             death_markers=[],
             n_bins=n_bins,
             save_dir=save_path)
    print(f"[ode_trajectories] Finished -> {save_path}")


# ------------------------------------------------------------------
def plot_ode(X, point_array, traj_array, death_markers, n_bins, save_dir):
    """Plot continuous trajectories and key time-points."""
    n_time = len(X)
    colors = plt.cm.get_cmap("viridis")(np.linspace(0.2, 0.98, n_time))

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'white',
        'axes.grid': False,
        'axes.labelcolor': 'black',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'font.size': 12,
    })

    fig, ax = plt.subplots(figsize=(10, 8), dpi=400)

    size, gap_size, bg_size = 80, (np.sqrt(80) * 1.1)**2, (np.sqrt(80) * 1.4)**2
    death_marker_size = 50

    # 1) Ground truth
    for t, data in enumerate(X):
        valid = data[~np.isnan(data).any(axis=1)]
        if len(valid) == 0:
            continue
        ax.scatter(valid[:, 0], valid[:, 1], c=[colors[t]] * len(valid),
                   marker='X', s=80, alpha=0.25, label=f'T{t}')

    # 2) Smooth trajectories
    n_traj = traj_array.shape[1]
    interp_num = 50
    for j in range(n_traj):
        traj = traj_array[:, j, :]
        valid_mask = ~np.isnan(traj).any(axis=1)
        valid_time = np.where(valid_mask)[0]
        valid_traj = traj[valid_mask]

        if 0 < len(valid_time) < n_bins and len(valid_traj) > 0:
            last_xy = valid_traj[-1]
            ax.scatter(last_xy[0], last_xy[1], c='white', s=death_marker_size,
                       marker='o', edgecolors='black', linewidths=1.5)

        if len(valid_time) < 3:
            if len(valid_time) >= 2:
                ax.plot(valid_traj[:, 0], valid_traj[:, 1], color='red',
                        alpha=0.75, linewidth=1.2, solid_capstyle='round')
            continue

        interp_t = np.linspace(valid_time.min(), valid_time.max(), interp_num)
        cs_x = CubicSpline(valid_time, valid_traj[:, 0], bc_type='natural')
        cs_y = CubicSpline(valid_time, valid_traj[:, 1], bc_type='natural')
        ax.plot(cs_x(interp_t), cs_y(interp_t), color='red',
                alpha=0.75, linewidth=1.2, solid_capstyle='round')

    # 3) Key points – same colors as ground truth
    gen_pts, gen_cols = [], []
    for t in range(n_time):
        pts = point_array[t, :, :]
        valid = ~np.isnan(pts).any(axis=1)
        if valid.sum() == 0:
            continue
        gen_pts.append(pts[valid])
        gen_cols.append(np.tile(colors[t], (valid.sum(), 1)))

    if gen_pts:
        all_pts = np.vstack(gen_pts)
        all_cols = np.vstack(gen_cols)
        ax.scatter(all_pts[:, 0], all_pts[:, 1], c='black', s=bg_size,
                   marker='o', linewidths=0)
        ax.scatter(all_pts[:, 0], all_pts[:, 1], c='white', s=gap_size,
                   marker='o', linewidths=0)
        ax.scatter(all_pts[:, 0], all_pts[:, 1], c=all_cols, s=size,
                   alpha=0.7, marker='o', linewidths=0)

    # 4) Legend
    legend_main = [
        Line2D([0], [0], marker='X', color='w', label='Ground Truth',
               markerfacecolor=colors[0], markersize=10, alpha=0.3),
        Line2D([0], [0], marker='o', color='w', label='Key Points',
               markerfacecolor=colors[-1], markersize=10),
        Line2D([0], [0], color='red', lw=2, label='Continuous Trajectory'),
        Line2D([0], [0], marker='o', color='w', label='Death Point',
               markerfacecolor='white', markeredgecolor='black', markersize=8)
    ]
    ax.legend(handles=legend_main, loc='upper right', frameon=True)

    legend_t = [Line2D([0], [0], marker='X', color='w',
                       label=f'DAY {t}', markerfacecolor=colors[t],
                       markersize=10, alpha=0.6) for t in range(n_time)]
    ax.legend(handles=legend_t, loc='upper left', title='Time Points', frameon=True)
    ax.add_artist(ax.get_legend())

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"Cell Trajectories ({n_traj} continuous differentiable trajectories)")

    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight')
    print(f"[plot_ode] Saved to -> {save_dir}")
    plt.close()


#%%
import anndata
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
from CytoBridge.tl.analysis import compute_velocity

def plot_velocity_stream(adata, save_dir, dim_reduction='none', device='cuda'):
    """
    Plot latent space velocity field streamlines
    Coordinates always use adata.obsm['X_latent'][:, :2]
    Velocity uses adata.layers['velocity_latent']
    If velocity is not 2-D, perform projection using its own basis
    """
    # 1. Ensure velocity has been calculated
    if 'velocity_latent' not in adata.layers:
        adata = compute_velocity(adata, device)

    # 2. Prepare coordinates (2-D)
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2)
        adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].copy()
    else:  # 'none' directly uses first two dimensions of latent space
        adata.obsm['X_umap'] = adata.obsm['X_latent'][:, :2].copy()

    # 3. Place velocity in scVelo's specified key
    adata.layers['Ms']   = adata.obsm['X_latent'].copy()
    adata.layers['velocity'] = adata.layers['velocity_latent']

    # 4. 2-D velocity processing
    if adata.layers['velocity'].shape[1] == 2:
        # Velocity is already 2-D, directly use as umap velocity
        adata.obsm['velocity_umap'] = adata.layers['velocity']
    else:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X')  # Calculate neighbors based on high-dim data
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)  # Build velocity graph from high-dim velocity vectors
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')  # Project 

    # 5. Time coloring
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])

    # 6. Plotting
    scv.settings.figdir = save_dir
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))

    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title='Velocity Stream Plot',
        legend_loc='right',
        palette='plasma',
        save='Velocity_Stream_Plot.svg',
    )
    print(f"Velocity stream plot saved to: {save_dir}/Velocity_Stream_Plot.svg")
    return adata