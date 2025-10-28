import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

#  TODO: add dim_reduction

def plot_growth(adata, dim_reduction='umap', save_path=None):
    if 'growth_rate' not in adata.obsm.keys():
        raise ValueError("Growth rate not found in adata.obsm. Please check if the growth term is set or the model is trained.")
    
    # Get dimensionality reduction coordinates for plotting
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm.keys():
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        plot_data = adata.obsm['X_umap']
    elif dim_reduction == 'PCA':
        plot_data = adata.obsm['X_pca'][:, :2]
    elif dim_reduction in ['none', None]:
        plot_data = adata.obsm['X_latent'][:, :2]
    else:
        raise ValueError(f"Invalid dim_reduction: {dim_reduction}")

    # Key modification: Reverse the order of points (last point becomes first, first point becomes last)
    plot_data = plot_data[::-1]  # Reverse coordinate order
    g_values = adata.obsm['growth_rate'][::-1]  # Synchronously reverse the order of color values (to ensure one-to-one correspondence)

    # Calculate color mapping range for growth rate
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

    # Plot scatter plot (now plotted in reversed order)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(plot_data[:, 0], plot_data[:, 1], c=colors, alpha=0.3, marker='o', s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array(g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Predicted Growth Rate')

    # Format color bar ticks
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{(x):.2f}'))

    # Save plot
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

    plt.show()

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from CytoBridge.utils.utils import load_model_from_adata
from CytoBridge.tl.analysis import generate_ode_trajectories
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.interpolate import make_interp_spline

def plot_ode(X, point_array, traj_array, save_dir):
    """
    Adapt to data containing NaNs: Stop drawing the subsequent part of the trajectory when NaN is encountered,
    and filter out circle points corresponding to NaNs.
    """
    num_timepoints = len(X)
    colors = plt.cm.get_cmap("viridis")(np.linspace(0, 1, num_timepoints))
    cmap = LinearSegmentedColormap.from_list("custom_viridis", colors)

    # Style settings
    sns.set_style("white")
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'axes.edgecolor': 'lightgrey',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.color': 'dimgrey',
        'ytick.color': 'dimgrey',
        'font.size': 12,
    })

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Point size formula
    size = 80
    point = np.sqrt(size)
    gap_size = (point + point * 0.1) ** 2
    bg_size = (np.sqrt(gap_size) + point * 0.3) ** 2

    # 1) Real observations (filter out NaNs in raw data)
    for t, data in enumerate(X):
        # Keep only samples without NaN in the current time step's raw data
        valid_data = data[~np.isnan(data).any(axis=1)]
        if len(valid_data) == 0:  # Skip this time step if no valid data
            continue
        ax.scatter(
            valid_data[:, 0], valid_data[:, 1],
            c=[colors[t]] * len(valid_data),
            marker='X', s=80, alpha=0.25,
            label=f"T{t}.0"
        )

    # 2) Trajectory (Key modification: Stop drawing subsequent parts when NaN is encountered)
    dense_T = 300  # Number of dense time points after interpolation
    num_trajectories = traj_array.shape[1]  # Total number of trajectories (number of particles)

    for j in range(num_trajectories):
        # Extract the complete trajectory of the j-th particle (time steps, 2D coordinates)
        traj = traj_array[:, j, :]
        # Find the position where the first NaN appears in the trajectory (previous parts are valid)
        # Check if each row contains NaN, return the index of the first True (return trajectory length if no NaN)
        first_nan_idx = np.argmax(np.isnan(traj).any(axis=1))
        # If no NaN exists, the valid segment is the entire trajectory; otherwise, take up to the first NaN
        if not np.isnan(traj).any(axis=1).any():
            valid_traj = traj
        else:
            valid_traj = traj[:first_nan_idx]  # Keep only the valid part before the first NaN
        
        # Skip this trajectory if the length of the valid segment is 0
        if len(valid_traj) < 2:  # At least 2 points are required for interpolation and plotting
            continue
        
        # Interpolate and smooth the valid segment (using only valid data)
        t_old = np.arange(len(valid_traj))  # Original time step indices (0,1,...,L-1, where L is valid length)
        t_new = np.linspace(0, t_old[-1], dense_T)  # Densified time points
        
        # Interpolate x and y coordinates (k=3 for cubic spline, requiring at least 4 points; use linear interpolation k=1 if insufficient)
        k_order = 3 if len(valid_traj) >= 4 else 1
        fx = make_interp_spline(t_old, valid_traj[:, 0], k=k_order)
        fy = make_interp_spline(t_old, valid_traj[:, 1], k=k_order)
        x_smooth = fx(t_new)
        y_smooth = fy(t_new)

        # Plot the valid segment of this trajectory
        ax.plot(x_smooth, y_smooth,
                color='red', alpha=0.75, linewidth=1.2, solid_capstyle='round')

    # 3) Generated points (Key modification: Filter out circle points corresponding to NaNs)
    # Extract generated points at all time steps and filter out points with NaNs
    gen_points_list = []  # Store all valid generated points
    gen_colors_list = []  # Store colors corresponding to valid points

    for t in range(point_array.shape[0]):  # Iterate over each time step
        # Extract all generated points at the t-th time step (number of particles, 2D)
        points_t = point_array[t, :, :]
        # Filter valid points without NaN at the current time step
        valid_mask_t = ~np.isnan(points_t).any(axis=1)
        valid_points_t = points_t[valid_mask_t]
        if len(valid_points_t) == 0:  # Skip this time step if no valid points
            continue
        # Colors corresponding to valid points (repeat the color of the current time step)
        valid_colors_t = np.tile(colors[t], (len(valid_points_t), 1))
        
        # Add to list
        gen_points_list.append(valid_points_t)
        gen_colors_list.append(valid_colors_t)

    # Plot three-layer circles only if there are valid generated points
    if gen_points_list:
        # Merge all valid generated points and their corresponding colors
        gen_points = np.concatenate(gen_points_list, axis=0)
        gen_colors = np.concatenate(gen_colors_list, axis=0)
        
        # Plot three-layer concentric circles (outer black background → middle white gap → inner colored points)
        ax.scatter(gen_points[:, 0], gen_points[:, 1],
                   c='black', s=bg_size, alpha=1, marker='o', linewidths=0)  # Outermost black background
        ax.scatter(gen_points[:, 0], gen_points[:, 1],
                   c='white', s=gap_size, alpha=1, marker='o', linewidths=0)  # Middle white gap
        ax.scatter(gen_points[:, 0], gen_points[:, 1],
                   c=gen_colors, s=size, alpha=0.7, marker='o', linewidths=0)  # Innermost colored points

    # 4) Legend (Handle duplicate labels possibly caused by NaN filtering)
    # Main legend (ground truth, predicted points, trajectories)
    legend_main = [
        Line2D([0], [0], marker='X', color='w', label='Ground Truth',
               markerfacecolor=colors[0], markersize=10, alpha=0.3),
        Line2D([0], [0], marker='o', color='w', label='Predicted',
               markerfacecolor=colors[-1], markersize=10),
        Line2D([0], [0], color='red', lw=2, label='Trajectory')
    ]
    ax.legend(handles=legend_main, loc='upper right', frameon=True)

    # Time step legend (only display time steps with valid data)
    legend_t = []
    for t in range(num_timepoints):
        # Check if the current time step has valid raw data (avoid displaying labels for time steps with no data)
        data_t = X[t]
        if len(data_t[~np.isnan(data_t).any(axis=1)]) > 0:
            legend_t.append(
                Line2D([0], [0], marker='X', color='w',
                       label=f'T{t}.0', markerfacecolor=colors[t], markersize=10, alpha=0.6)
            )
    if legend_t:  # Add time step legend only if there are valid time steps
        ax.legend(handles=legend_t, loc='upper left', title='Time Points', frameon=True)
        ax.add_artist(ax.get_legend())  # Keep the main legend

    ax.set_xlabel("Gene $X_1$")
    ax.set_ylabel("Gene $X_2$")
    ax.set_title("Real vs Generated Samples & Trajectories")

    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight')  # Use bbox_inches to avoid legend being cut off
    print(f"[plot_ode_trajectories] saved to -> {save_dir}")
    plt.close()

# ---------- Parent function: responsible for dimensionality reduction + plotting ----------
import os
import pickle
import numpy as np

def plot_ode_trajectories(
    adata,
    save_dir='figures',
    save_base_name='ode_trajectories.pdf',
    n_trajectories=50,
    n_bins=100,
    dim_reduction='umap',
    device="cuda",
):
    model = load_model_from_adata(adata)
    exp_dir = os.path.join(save_dir, "ode_results")
    os.makedirs(exp_dir, exist_ok=True)

    point_array, traj_array = generate_ode_trajectories(
        model, adata,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
    )

    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())
    X_raw = adata.obsm['X_latent']
    D = traj_array.shape[-1]

    
    if D == 2:
        traj_2d, point_2d = traj_array, point_array
        X_2d = X_raw
    else:
        reducer_pkl = os.path.join(exp_dir, "dim_reducer.pkl")

        if dim_reduction == 'none':
            traj_2d  = traj_array[..., :2]
            point_2d = point_array[..., :2]
            X_2d     = X_raw[..., :2]
        elif dim_reduction in ('pca', 'umap'):
            if os.path.isfile(reducer_pkl):
                with open(reducer_pkl, 'rb') as f:
                    reducer = pickle.load(f)
                print(f"[dim-reduction] loaded existing reducer from {reducer_pkl}")
                # For safety, check the dimension
                if hasattr(reducer, 'n_components') and reducer.n_components != 2:
                    raise RuntimeError("Cached reducer n_components != 2, please delete it manually.")
            else:
                print(f"[dim-reduction] no cached reducer, training {dim_reduction.upper()} ...")
                if dim_reduction == 'pca':
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2, random_state=0)
                else:  # umap
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=0)
                reducer.fit(X_raw)
                
                with open(reducer_pkl, 'wb') as f:
                    pickle.dump(reducer, f)
                print(f"[dim-reduction] reducer saved to {reducer_pkl}")

            
            X_2d     = reducer.transform(X_raw)
            traj_2d  = reducer.transform(traj_array.reshape(-1, D)).reshape(*traj_array.shape[:2], 2)
            point_2d = reducer.transform(point_array.reshape(-1, D)).reshape(*point_array.shape[:2], 2)
        else:
            raise ValueError("dim_reduction must be 'none', 'pca' or 'umap'")

        # Slice by time
    X = [X_2d[adata.obs[samples_key] == t] for t in unique_times]

    # 3. Save & plot
    np.save(os.path.join(exp_dir, "ode_traj.npy"),  traj_2d)
    np.save(os.path.join(exp_dir, "ode_point.npy"), point_2d)
    print(f"[generate_ode_trajectories] trajectories saved to {exp_dir}")

    save_path = os.path.join(exp_dir, save_base_name)
    plot_ode(X=X,
             point_array=point_2d,
             traj_array=traj_2d,
             save_dir=save_path)

    print(f"[ode_trajectories] done -> {save_path}")

#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
import CytoBridge
import scanpy as sc
from matplotlib import rcParams
import math
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    'axes.spines.right': False,
    'axes.spines.top': False,
    'font.size': 12,
    'axes.labelsize': 12
})

def plot_score_and_gradient(dynamical_model, device, t_value=1.0, x_range=(0, 2), y_range=(0, 2.5),
                            step=5, save_path=None, cmap='rainbow'):
    """
    Plot the output log probability of the score component in DynamicalModel and its gradient vector field (adapted for scoreNet2)
    """
    device = torch.device(device)          # Ensure it's a torch.device object
    dynamical_model = dynamical_model.to(device)   # ← Critical: Move model to device
    if 'score' not in dynamical_model.components:
        raise ValueError("DynamicalModel must contain 'score' component")

    if dynamical_model.velocity_net.input_layer[0].in_features !=3:
        raise ValueError("DynamicalModel's input (gene expreesion) expression must be 2")
        
    # Create grid (100x100 grid)
    grid_size = 100
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    xv, yv = np.meshgrid(x, y)
    grid_points = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # Shape: (10000, 2)

    # Convert to tensor (separate time and position, adapted to input format of scoreNet2)

    x_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)
    t_value = torch.tensor([t_value], dtype=torch.float32, device=device).unsqueeze(1)
    log_density, gradients = dynamical_model.compute_score(t_value, x_tensor)
    # Convert to numpy and reshape to grid shape
    log_density_np = log_density.cpu().detach().numpy().reshape(grid_size, grid_size)  # 100x100
    density = torch.exp(log_density).cpu().detach().numpy().reshape(grid_size, grid_size)
    gradients_np = gradients.cpu().detach().numpy().reshape(grid_size, grid_size, 2)  # 100x100x2

    # Sample gradient vectors (downsample to avoid overcrowding)
    xv_quiver = xv[::step, ::step]
    yv_quiver = yv[::step, ::step]
    grad_quiver = gradients_np[::step, ::step, :]  # Sampled gradients


    # Plotting
    plt.figure(figsize=(8, 6))
    # Plot negative log probability contours
    contour = plt.contourf(
        xv, yv, -log_density_np,  # Use negative log probability
        levels=50,
        cmap=cmap,
        alpha=0.8
    )
    plt.colorbar(contour, label='-log density')

    # Plot gradient vector field
    plt.quiver(
        xv_quiver, yv_quiver,
        grad_quiver[:, :, 0], grad_quiver[:, :, 1],
        color='white',
        scale=1,
    )

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f't = {t_value}')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    else:
        plt.show()
    plt.close()


#%%
import os
import math
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from torch.nn import Module
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from CytoBridge.tl.analysis import generate_sde_trajectories


def plot_sde_trajectories(
        adata: sc.AnnData,
        exp_dir: str = "./sde_results",
        device: str = "cuda",
        sigma: float = 1.0,
        n_time_steps: int = 100,
        sample_traj_num: int = 100,
        sample_cell_tra_consistent: bool = False,  # Controls whether trajectories and predicted points share the same initial cells
        init_time: int = 0,
        dim_reduction: str = 'none'
):
    """
    Generate SDE trajectory plots from a CytoBridge-trained AnnData object.
    
    When sample_cell_tra_consistent=True: Only plots predicted points from the same initial cells used for trajectories,
    resulting in fewer predicted points that are directly linked to visualized trajectories.
    
    When sample_cell_tra_consistent=False: Maintains original behavior - plots all predicted points 
    while using a subset for trajectories.
    
    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing pre-trained CytoBridge model and latent features (X_latent).
    exp_dir : str, default="./sde_results"
        Directory to save SDE results and plots.
    device : str, default="cuda"
        Device for computation ("cuda" for GPU, "cpu" for CPU).
    sigma : float, default=1.0
        Diffusion coefficient for the SDE.
    n_time_steps : int, default=100
        Number of continuous time steps for SDE simulation.
    sample_traj_num : int, default=100
        Number of trajectories to sample for visualization (prevents overplotting).
    sample_cell_tra_consistent : bool, default=False
        If True, uses identical initial cells for both trajectories and predicted points.
    init_time : int, default=0
        Initial time point to start SDE simulation (must exist in adata.obs).
    dim_reduction : str, default='none'
        Dimensionality reduction method for 2D plotting:
        - 'none': Use first 2 dimensions of X_latent directly
        - 'pca': Project via precomputed PCA (requires adata.obsm['X_pca'])
        - 'umap': Project via precomputed UMAP (auto-computes if missing)
    
    Returns
    -------
    sc.AnnData
        Original AnnData object (unchanged).
    """
    # ---------- Initialization ----------
    os.makedirs(exp_dir, exist_ok=True)
    device = torch.device(device)
    print(f"Using device: {device}")



    sde_traj, sde_point, w_point = generate_sde_trajectories(
        adata,
        exp_dir=exp_dir,
        device=device,
        sigma=sigma,
        n_time_steps=n_time_steps,
        sample_traj_num=sample_traj_num,
        sample_cell_tra_consistent=sample_cell_tra_consistent,
        init_time=init_time,
    )

    # ---------- Project to 2D Space (Plot Preparation) ----------
    dim_reduction = str(dim_reduction).lower()
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        plot_2d = adata.obsm['X_umap'][:, :2]
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            raise ValueError('Please run sc.pp.pca(adata) first to compute PCA embeddings')
        plot_2d = adata.obsm['X_pca'][:, :2]
    elif dim_reduction in {'none', 'null', ''}:
        plot_2d = adata.obsm['X_latent'][:, :2]
    else:
        raise ValueError(f"Invalid dim_reduction method: {dim_reduction}. Choose 'none', 'pca', or 'umap'")
    
    time_key = 'time_point_processed'
    real_df = pd.DataFrame(plot_2d, columns=['x1', 'x2'])
    real_df['samples'] = adata.obs[time_key].values

    # Project trajectories/discrete points to the same 2D space
    if dim_reduction == 'pca':
        mean_ = adata.uns['pca']['mean'] if 'mean' in adata.uns['pca'] else 0.
        comps = adata.uns['pca']['components_'][:, :2]
        traj_2d = np.dot(sde_traj - mean_, comps)  # Shape: [n_steps, sample_traj_num, 2]
        point_2d = np.dot(sde_point - mean_, comps)  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)
    elif dim_reduction == 'umap':
        raise NotImplementedError(
            "UMAP projection for SDE results requires offline transform of saved sde_traj/sde_point. "
            "Use 'none' or 'pca' mode instead."
        )
    else:  # 'none': Use first 2 dimensions of latent space directly
        traj_2d = sde_traj[..., :2]  # Shape: [n_steps, sample_traj_num, 2]
        point_2d = sde_point[..., :2]  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)

    # -------------------------- Key Modification 2: Updated Plotting Input --------------------------
    # Generate SDE Trajectory Plot
    print("Plotting SDE trajectories...")
    sde_plot(df=real_df, 
             generated=point_2d,  # point_2d now contains only cells matching trajectories when consistent=True
             trajectories=traj_2d,
             save=True, 
             path=exp_dir, 
             file='sde_trajectories.pdf')

    return adata


# ------------------------------------------------------------------------------
# Plotting Function Dependencies: Matplotlib, Linear Segmented Colormap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np  # Re-import for plotting (safe for modularity)


def sde_plot(df, generated, trajectories, palette='viridis',
             df_time_key='samples', x='x1', y='x2',
             save=False, path="./sde_results", file='sde_trajectories.pdf'):
    """
    Plot SDE trajectories with ground truth data, predicted points, and trajectory lines.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of ground truth data with columns for x/y coordinates and time.
    generated : np.ndarray
        SDE-predicted discrete points (shape: [n_time_points, batch_size, 2]).
    trajectories : np.ndarray
        SDE-simulated continuous trajectories (shape: [n_time_steps, batch_size, 2]).
    palette : str, default='viridis'
        Matplotlib colormap name for time-based coloring.
    df_time_key : str, default='samples'
        Column name in df containing time labels.
    x : str, default='x1'
        Column name in df for x-axis coordinates.
    y : str, default='x2'
        Column name in df for y-axis coordinates.
    save : bool, default=False
        Whether to save the plot to file.
    path : str, default="./sde_results"
        Directory to save the plot (only used if save=True).
    file : str, default='sde_trajectories.pdf'
        Filename for the saved plot (only used if save=True).
    
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib Figure object of the SDE trajectory plot.
    """
    # Get sorted unique time groups
    groups = sorted(df[df_time_key].unique())
    # Initialize custom colormap
    cmap = plt.colormaps.get_cmap(palette)
    new_cmap = LinearSegmentedColormap.from_list('viridis_custom',
                                                 cmap(np.linspace(0, 1, 256)))

    # Configure plot aesthetics
    plt.rcParams.update({
        'axes.edgecolor': 'lightgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'xtick.bottom': False,
        'ytick.left': False,
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 10
    })

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)

    # 1. Plot ground truth data points (X markers)
    ax.scatter(df[x], df[y], c=df[df_time_key], s=100, alpha=0.5,
               marker='X', cmap=new_cmap, label='Ground Truth')

    # 2. Plot SDE trajectories (thin red lines with low opacity to avoid overcrowding)
    # Transpose trajectories to (batch_size, n_time_steps, 2) for per-trajectory iteration
    for traj in np.transpose(trajectories, (1, 0, 2)):
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.4, color='firebrick', lw=1)

    # 3. Plot SDE-predicted discrete points (O markers with double-layer coloring for contrast)
    pred_points = np.concatenate(generated, axis=0)  # Merge predicted points across all time steps
    # Match time labels to predicted points (repeat time for all points in the same time step)
    pred_times = np.concatenate([[t] * generated[i].shape[0]
                                 for i, t in enumerate(groups)])
    # Outer black layer (enhances visibility against background)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c='black',
               s=120, alpha=0.8, marker='o')
    # Inner colored layer (encodes time information)
    ax.scatter(pred_points[:, 0], pred_points[:, 1], c=pred_times,
               s=100, alpha=0.8, marker='o', cmap=new_cmap, label='SDE Predicted')

    # 4. Build legends (time labels + data type)
    # Legend for time points (top-left)
    time_legend = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=new_cmap(i / len(groups)), markersize=12)
                   for i, t in enumerate(groups)]
    leg1 = ax.legend(handles=time_legend, loc='upper left', title='Time Points')
    ax.add_artist(leg1)  # Preserve time legend (avoid being overwritten by subsequent legends)

    # Legend for data types (top-right)
    type_legend = [
        Line2D([0], [0], marker='X', color='w', label='Ground Truth',
               markerfacecolor='grey', markersize=12, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='SDE Predicted',
               markerfacecolor='grey', markersize=12),
        Line2D([0], [0], color='red', lw=2, label='SDE Trajectory', alpha=0.5)
    ]
    ax.legend(handles=type_legend, loc='upper right')

    # 5. Set axis labels and plot title
    ax.set_xlabel("Latent Dimension 1", fontsize=14)
    ax.set_ylabel("Latent Dimension 2", fontsize=14)
    ax.set_title("SDE Trajectories (Combining Velocity/Growth/Score)", fontsize=16, pad=20)

    # 6. Save plot if enabled (adjust layout to prevent label cutoff)
    if save:
        plt.tight_layout()
        # Create directory if it doesn't exist (defensive check)
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, file), bbox_inches='tight')
        print(f"SDE trajectory plot saved to: {os.path.join(path, file)}")

    # Close plot to free memory (avoids unintended memory leaks in batch runs)
    plt.close()

    return fig

#%%
import anndata
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
from CytoBridge.tl.analysis import compute_velocity

def plot_velocity_stream(adata, output_path, dim_reduction='none', device='cuda'):
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
    scv.settings.figdir = output_path
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
    print(f"Velocity stream plot saved to: {output_path}/Velocity_Stream_Plot.svg")
    return adata
#%%
import anndata
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import torch

def plot_score_stream(adata, output_path, dim_reduction='none', device='cuda'):
    """
    Plot score function diffusion velocity field streamlines
    Coordinates use first two dimensions of adata.obsm['X_latent'] or dimensionality reduction results
    Velocity is calculated from the gradient of the score function (log density)
    """
    # 1. Check if necessary data exists
    required_keys = ['X_latent', 'time_point_processed']
    for key in required_keys:
        if key not in adata.obsm and key not in adata.obs:
            raise ValueError(f"Required data not found: {key}, please ensure the model has correctly computed relevant results")

    # 2. Prepare input data (latent space data and time)
    device = torch.device(device)
    all_data = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    all_data.requires_grad_(True)  # Enable gradient tracking (critical modification)
    all_times = torch.tensor(adata.obs['time_point_processed'].values, dtype=torch.float32, device=device).unsqueeze(1)

    # 3. Calculate score function (log density) and gradients (velocity vectors)
    # Calculate log density values directly from the model, ensuring complete computation graph
    model = load_model_from_adata(adata)  # Load model (critical modification)
    model.to(device)
    model.eval()
    log_density_values, gradients = model.compute_score(all_times, all_data)  # Directly compute with model (critical modification)

    # 4. Process coordinate and velocity dimensionality reduction
    data_np = all_data.detach().cpu().numpy()
    gradients_np = gradients.detach().cpu().numpy()

    # Prepare 2D coordinates
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_latent')
            sc.tl.umap(adata)
        data_np_2d = adata.obsm['X_umap']
        # Perform same dimensionality reduction on gradients
        data_end = data_np + gradients_np
        from sklearn.manifold import umap
        reducer = umap.UMAP().fit(adata.obsm['X_latent'])  # Reuse trained UMAP
        data_end_2d = reducer.transform(data_end)
        gradients_np_2d = data_end_2d - data_np_2d
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2, use_rep='X_latent')
        data_np_2d = adata.obsm['X_pca'][:, :2]
        # Perform same dimensionality reduction on gradients
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(adata.obsm['X_latent'])
        data_end = data_np + gradients_np
        data_end_2d = pca.transform(data_end)
        gradients_np_2d = data_end_2d - data_np_2d
    else:  # Directly use first two dimensions of latent space
        data_np_2d = data_np[:, :2]
        gradients_np_2d = gradients_np[:, :2]

    # 5. Normalize velocity vectors (uniform length for visualization)
    gradients_np_2d = gradients_np_2d / np.linalg.norm(gradients_np_2d, axis=1, keepdims=True) * 5

    # 6. Prepare data format required by scVelo
    adata.obsm['X_umap'] = data_np_2d  # 2D coordinates (compatible with scVelo's basis='umap')
    adata.layers['Ms'] = data_np  # Original high-dimensional state matrix
    adata.layers['velocity'] = gradients_np  # High-dimensional velocity vectors
    if gradients_np_2d.shape[1] == 2:
        adata.obsm['velocity_umap'] = gradients_np_2d  # Directly use 2D velocity
    else:
        # High-dimensional velocity needs neighbor graph construction and projection to 2D coordinates
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_latent')
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')

    # 7. Time information processing
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])

    # 8. Plot settings and saving
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))

    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title='Score Stream Plot',
        legend_loc='right',
        palette='plasma',
        save='Score_Stream_Plot.svg',
    )
    print(f"Score function diffusion velocity stream plot saved to: {output_path}/Score_Stream_Plot.svg")
    return adata
#%%
import anndata
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from CytoBridge.utils.utils import load_model_from_adata  # Ensure correct import

def plot_combined_velocity_stream(adata, output_path, dim_reduction='none', device='cuda'):
    """
    Plot combined velocity stream plot (model velocity + score gradient velocity)
    
    Parameters:
        - alpha: weight for model velocity
        - beta: weight for score gradient velocity
    """
    alpha=0.5
    beta=0.5
    # --------------------------
    # 1. Calculate and obtain both velocities
    # --------------------------
    # 1.1 Obtain model velocity (velocity_latent)
    if 'velocity_latent' not in adata.layers:
        from CytoBridge.tl.analysis import compute_velocity
        adata = compute_velocity(adata, device)
    vel = adata.layers['velocity_latent'].copy()  # Model velocity
    norms = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_model = np.divide(vel, norms, out=np.zeros_like(vel), where=norms!=0) * 5
    # 1.2 Calculate score gradient velocity
    device = torch.device(device)
    model = load_model_from_adata(adata)
    model.to(device)
    model.eval()
    
    # Prepare input data
    all_data = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    all_data.requires_grad_(True)
    all_times = torch.tensor(
        adata.obs['time_point_processed'].values, 
        dtype=torch.float32, 
        device=device
    ).unsqueeze(1)
    
    # Calculate score gradient (velocity)
    _, grad_score = model.compute_score(all_times, all_data)
    vel_score1 = grad_score.detach().cpu().numpy()  # Score gradient velocity
    norms = np.linalg.norm(vel_score1, axis=1, keepdims=True)
    vel_score = np.divide(vel_score1, norms, out=np.zeros_like(vel), where=norms!=0)
    # --------------------------
    # 2. Combine both velocities (weighted sum, maintaining original calculation method)
    # --------------------------
    combined_vel = alpha * vel_model + beta * vel_score  # Consistent with overall calculation logic in example
    adata.layers['combined_velocity'] = combined_vel  # Store combined velocity

    # --------------------------
    # 3. Prepare coordinates (2D), maintaining same dimensionality reduction logic as existing plotting functions
    # --------------------------
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_latent')
            sc.tl.umap(adata)
        adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2, use_rep='X_latent')
        adata.obsm['X_umap'] = adata.obsm['X_pca'][:, :2].copy()
    else:  # Directly use first two dimensions of latent space
        adata.obsm['X_umap'] = adata.obsm['X_latent'][:, :2].copy()

    # --------------------------
    # 4. Adapt to scVelo format, maintaining same projection logic as existing functions
    # --------------------------
    adata.layers['Ms'] = adata.obsm['X_latent'].copy()  # Original high-dimensional state
    adata.layers['velocity'] = adata.layers['combined_velocity']  # Combined velocity

    # Process velocity projection (judge based on dimensions, consistent with logic in plot_velocity_stream/plot_score_stream)
    if combined_vel.shape[1] == 2:
        adata.obsm['velocity_umap'] = combined_vel  # Directly use 2D velocity
    else:
        # High-dimensional velocity needs neighbor graph construction and projection to 2D
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_latent')
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')

    # --------------------------
    # 5. Plot settings, maintaining same style as existing functions
    # --------------------------
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=300, figsize=(7, 5))

    # Plot combined velocity stream
    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title=f'Combined Velocity Stream',
        legend_loc='right',
        palette='plasma',
        save='All_Velocity_Stream.svg',
    )
    print(f"Combined velocity stream plot saved to: {output_path}/All_Velocity_Stream.svg")
    return adata
