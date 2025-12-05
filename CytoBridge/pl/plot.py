import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

# TODO: add dim_reduction

def plot_growth(adata, dim_reduction='umap', output_path=None):
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
    g_values = adata.obsm['growth_rate'][::-1]  # Synchronously reverse color value order (ensure one-to-one correspondence)

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
    fig, ax = plt.subplots(figsize=(6, 4))
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
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', transparent=True)
        print(f"[plot_growth] saved to -> {output_path}")
    plt.show()

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from CytoBridge.tl.analysis import generate_ode_trajectories
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.interpolate import make_interp_spline

def plot_ode(X, point_array, traj_array, save_dir):
    """
    Adapt to data containing NaNs: Stop plotting subsequent parts of the trajectory when NaN is encountered; filter out circle points corresponding to NaNs
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

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    # Point size formula
    size = 80
    point = np.sqrt(size)
    gap_size = (point + point * 0.1) ** 2
    bg_size = (np.sqrt(gap_size) + point * 0.3) ** 2

    # 1) Real observations (filter out NaNs in original data)
    for t, data in enumerate(X):
        # Keep only samples without NaN in the current time step's original data
        valid_data = data[~np.isnan(data).any(axis=1)]
        if len(valid_data) == 0:  # Skip this time step if no valid data
            continue
        ax.scatter(
            valid_data[:, 0], valid_data[:, 1],
            c=[colors[t]] * len(valid_data),
            marker='X', s=80, alpha=0.25,
            label=f"T{t}.0"
        )

    # 2) Trajectory (Key modification: Stop plotting subsequent parts of the trajectory when NaN is encountered)
    dense_T = 300  # Number of dense time points after interpolation
    num_trajectories = traj_array.shape[1]  # Total number of trajectories (number of particles)

    for j in range(num_trajectories):
        # Extract the complete trajectory of the j-th particle (time steps, 2D coordinates)
        traj = traj_array[:, j, :]
        # Find the position where the first NaN appears in the trajectory (all before are valid segments)
        # Check if each row contains NaN, return the index of the first True (return trajectory length if no NaN)
        first_nan_idx = np.argmax(np.isnan(traj).any(axis=1))
        # If there is no NaN, the valid segment is the entire trajectory; otherwise, take up to the first NaN
        if not np.isnan(traj).any(axis=1).any():
            valid_traj = traj
        else:
            valid_traj = traj[:first_nan_idx]  # Keep only valid parts before the first NaN
        
        # Skip this trajectory if the length of the valid segment is 0
        if len(valid_traj) < 2:  # At least 2 points are needed for interpolation and plotting
            continue
        
        # Interpolate and smooth the valid segment (use only valid data)
        t_old = np.arange(len(valid_traj))  # Original time step indices (0,1,...,L-1, where L is valid length)
        t_new = np.linspace(0, t_old[-1], dense_T)  # Densified time points
        
        # Interpolate x and y coordinates (k=3 for cubic spline, requires at least 4 points; use linear interpolation k=1 if insufficient)
        k_order = 3 if len(valid_traj) >= 4 else 1
        fx = make_interp_spline(t_old, valid_traj[:, 0], k=k_order)
        fy = make_interp_spline(t_old, valid_traj[:, 1], k=k_order)
        x_smooth = fx(t_new)
        y_smooth = fy(t_new)

        # Plot the valid segment of this trajectory
        ax.plot(x_smooth, y_smooth,
                color='red', alpha=0.75, linewidth=1.2, solid_capstyle='round')

    # 3) Generated points (Key modification: Filter out circle points corresponding to NaNs)
    # Extract generated points at all time steps and filter out points containing NaNs
    gen_points_list = []  # Store all valid generated points
    gen_colors_list = []  # Store colors corresponding to valid points

    for t in range(point_array.shape[0]):  # Iterate over each time step
        # Extract all generated points at the t-th time step (number of particles, 2)
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

    # 4) Legend (Handle duplicate labels that may result from NaN filtering)
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
        # Check if the current time step has valid original data (avoid displaying labels for time steps with no data)
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
    plt.savefig(save_dir, bbox_inches='tight')  # bbox_inches prevents legend from being cut off
    print(f"[plot_ode_trajectories] saved to -> {save_dir}")
    plt.show()
    #plt.close()

# ---------- Parent function: responsible for dimensionality reduction + plotting ----------
import os
import pickle
import numpy as np

def plot_ode_trajectories(
    adata,
    model,
    output_path='figures',
    save_base_name='ode_trajectories.pdf',
    n_trajectories=50,
    n_bins=100,
    dim_reduction='umap',
    device="cuda",
    split_true=False,
):
    model.to(device)
    model.eval()
    output_path = os.path.join(output_path, "ode_results")
    os.makedirs(output_path, exist_ok=True)

    point_array, traj_array = generate_ode_trajectories(
        model, adata,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        device=device,
        split_true=split_true
    )

    samples_key = 'time_point_processed'
    unique_times = np.sort(adata.obs[samples_key].unique())
    X_raw = adata.obsm['X_latent']
    D = traj_array.shape[-1]

    # 2. Reduce dimension to 2-D
    if D == 2:
        traj_2d, point_2d = traj_array, point_array
        X_2d = X_raw
    else:
        reducer_pkl = os.path.join(output_path, "dim_reducer.pkl")

        if dim_reduction == 'none':
            traj_2d  = traj_array[..., :2]
            point_2d = point_array[..., :2]
            X_2d     = X_raw[..., :2]
        elif dim_reduction in ('pca', 'umap'):
            # --- First check if there is an existing reducer locally ---
            if os.path.isfile(reducer_pkl):
                with open(reducer_pkl, 'rb') as f:
                    reducer = pickle.load(f)
                print(f"[dim-reduction] loaded existing reducer from {reducer_pkl}")
                # For safety, check dimension
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
                # Save for future reuse
                with open(reducer_pkl, 'wb') as f:
                    pickle.dump(reducer, f)
                print(f"[dim-reduction] reducer saved to {reducer_pkl}")

            # --- Uniform dimensionality reduction ---
            X_2d     = reducer.transform(X_raw)
            traj_2d  = reducer.transform(traj_array.reshape(-1, D)).reshape(*traj_array.shape[:2], 2)
            point_2d = reducer.transform(point_array.reshape(-1, D)).reshape(*point_array.shape[:2], 2)
        else:
            raise ValueError("dim_reduction must be 'none', 'pca' or 'umap'")

        # Slice by time
    X = [X_2d[adata.obs[samples_key] == t] for t in unique_times]

    # 3. Save & plot
    np.save(os.path.join(output_path, "ode_traj.npy"),  traj_2d)
    np.save(os.path.join(output_path, "ode_point.npy"), point_2d)
    print(f"[generate_ode_trajectories] trajectories saved to {output_path}")

    output_path = os.path.join(output_path, save_base_name)
    plot_ode(X=X,
             point_array=point_2d,
             traj_array=traj_2d,
             save_dir=output_path)

    print(f"[ode_trajectories] done -> {output_path}")
    

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
                            step=5, output_path=None, cmap='rainbow'):
    """
    Plot the output log probability of the score component in DynamicalModel and its gradient vector field (adapted for scoreNet2)
    """
    device = torch.device(device)          # Ensure it's a torch.device object
    dynamical_model = dynamical_model.to(device)   # ← Critical: Move model to device
    if 'score' not in dynamical_model.components:
        raise ValueError("DynamicalModel must contain 'score' component")

    if dynamical_model.velocity_net.input_layer[0].in_features !=3:
        raise ValueError("DynamicalModel's input (gene expression) must be 2-dimensional")
        
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

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
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
        output_path: str = "./sde_results",
        device: str = "cuda",
        sigma: float = 1.0,
        n_bins: int = 10,
        n_trajectories: int = 100,
        init_time: int = 0,
        dim_reduction: str = 'none',
        split_true: bool =False

):
    """
    Generate SDE trajectory plots from a CytoBridge-trained AnnData object.
    

    
    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing pre-trained CytoBridge model and latent features (X_latent).
    output_path : str, default="./sde_results"
        Directory to save SDE results and plots.
    device : str, default="cuda"
        Device for computation ("cuda" for GPU, "cpu" for CPU).
    sigma : float, default=1.0
        Diffusion coefficient for the SDE.
    n_bins : int, default=100
        Number of continuous time steps for SDE simulation.
    n_trajectories : int, default=100
        Number of trajectories to sample for visualization (prevents overplotting).

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
    os.makedirs(output_path, exist_ok=True)
    device = torch.device(device)
    print(f"Using device: {device}")


    sde_traj, sde_point, w_point = generate_sde_trajectories(
        adata,
        exp_dir=output_path,
        device=device,
        sigma=sigma,
        n_time_steps=n_bins,
        sample_traj_num=n_trajectories,
        init_time=init_time,
        split_true=split_true
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
    if (sde_traj.shape[-1]==2):
        traj_2d = sde_traj
        point_2d = sde_point[..., :2]  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)

    elif dim_reduction == 'pca':
        mean_ = adata.uns['pca']['mean'] if 'mean' in adata.uns['pca'] else 0.
        comps = adata.uns['pca']['components_'][:, :2]
        traj_2d = np.dot(sde_traj - mean_, comps)  # Shape: [n_steps, sample_traj_num, 2]
        point_2d = np.dot(sde_point - mean_, comps)  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)
    elif dim_reduction == 'umap':

        from umap import UMAP       
        reducer = UMAP().fit(adata.obsm['X_latent'])

        def flatten_and_2d(model, x):
            """
            x: 任意维，只要最后一维是特征
            返回: 与 x 同形但最后一维变为 2 的数组
            """
            *lead, F = x.shape               # 前面所有维度 + 特征维
            x_2d = x.reshape(-1, F)          # (n_samples, F)
            y_2d = model.transform(x_2d)     # (n_samples, 2)
            return y_2d.reshape(*lead, 2)    # 原维度拼回，最后维=2

        # ---------- 一次性处理 ----------
        traj_2d = flatten_and_2d(reducer, sde_traj)   # (n_time, n_traj, 2)
        point_2d = flatten_and_2d(reducer, sde_point) # (n_time, n_traj, 2)
        
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
             output_path=output_path, 
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
             save=False, output_path="./sde_results", file='sde_trajectories.pdf'):
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
    output_path : str, default="./sde_results"
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)

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
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(os.path.join(output_path, file), bbox_inches='tight')
        print(f"SDE trajectory plot saved to: {os.path.join(output_path, file)}")
    fig.show()

    # Close plot to free memory (avoids unintended memory leaks in batch runs)
    #plt.close()

    return fig

#%%
import anndata
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
from CytoBridge.tl.analysis import compute_velocity

def plot_velocity_stream(adata, model,output_path, dim_reduction='none', device='cuda',color_key = None):
    """
    Plot latent space velocity field streamlines
    Coordinates always use adata.obsm['X_latent'][:, :2]
    Velocity uses adata.layers['velocity_latent']
    If velocity is not 2-D, perform projection using its own basis
    """
    # 1. Ensure velocity has been calculated
    if 'velocity_latent' not in adata.layers:
        adata = compute_velocity(adata, model,device)

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
    scv.settings.set_figure_params(dpi=100, figsize=(7, 5))

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
    if color_key != None:
        scv.pl.velocity_embedding_stream(
            adata,
            basis='umap',
            color='cluster',
            figsize=(7, 5),
            density=3,
            title='Velocity Stream Plot',
            legend_loc='right',
            palette='plasma',
            save='Velocity_cluster_Stream_Plot.svg',
        )
    return adata
#%%
import anndata
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from CytoBridge.tl.analysis import compute_interaction_force  # Import interaction calculation function

def plot_interaction_stream(adata, output_path, dim_reduction='none', device='cuda'):
    """
    Plot interaction force field streamlines
    Coordinates use first two dimensions of adata.obsm['X_latent'] or dimensionality reduction results
    Force vectors are calculated from the model's interaction component
    """
    # 1. Ensure interaction force has been calculated
    if 'interaction_force' not in adata.layers:
        adata = compute_interaction_force(adata, device)
    
    # 2. Check required data
    required_keys = ['X_latent', 'time_point_processed']
    for key in required_keys:
        if (key not in adata.obsm) and (key not in adata.obs):
            raise ValueError(f"Required data not found: {key}")
    
    # 3. Prepare 2D coordinates
    if dim_reduction == 'umap':
        if 'X_umap' not in adata.obsm:
            sc.pp.neighbors(adata, use_rep='X_latent')
            sc.tl.umap(adata)
        coords_2d = adata.obsm['X_umap']
    elif dim_reduction == 'pca':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=2, use_rep='X_latent')
        coords_2d = adata.obsm['X_pca'][:, :2]
    else:  # Use first two dimensions of latent space
        coords_2d = adata.obsm['X_latent'][:, :2].copy()
    
    # 4. Prepare interaction force vectors
    force = adata.layers['interaction_force']
    # print(force.shape)
    # print(force)
    # 5. Process force vectors for 2D visualization
    if force.shape[1] == 2:
        # Force is already 2D
        force_2d = force
    else:
        # Project high-dimensional force to 2D using the same reduction
        if dim_reduction == 'umap':
            from umap import UMAP       
            reducer = UMAP().fit(adata.obsm['X_latent'])
            data_end = adata.obsm['X_latent'] + force
            data_end_2d = reducer.transform(data_end)
            force_2d = data_end_2d - coords_2d
        elif dim_reduction == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2).fit(adata.obsm['X_latent'])
            data_end = adata.obsm['X_latent'] + force
            data_end_2d = pca.transform(data_end)
            force_2d = data_end_2d - coords_2d
        else:
            # Directly use first two dimensions for 'none'
            force_2d = force[:, :2]

    # 6. Normalize force vectors for visualization
    force_2d = force_2d / np.linalg.norm(force_2d, axis=1, keepdims=True) * 5
    
    # 7. Prepare scVelo required format
    adata.obsm['X_umap'] = coords_2d
    adata.layers['Ms'] = adata.obsm['X_latent'].copy()
    adata.layers['velocity'] = force  # Use interaction force as "velocity" for scVelo
    # Handle high-dimensional force projection if needed
    if force_2d.shape[1] == 2:
        adata.obsm['velocity_umap'] = force_2d
    else:
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_latent')
        scv.tl.velocity_graph(adata, vkey='velocity', n_jobs=16)
        scv.tl.velocity_embedding(adata, basis='umap', vkey='velocity')
    
    # 8. Time information for coloring
    adata.obs['time_categorical'] = pd.Categorical(adata.obs['time_point_processed'])
    
    # 9. Plot settings
    scv.settings.figdir = output_path
    scv.settings.set_figure_params(dpi=100, figsize=(7, 5))


    # 10. Generate and save plot
    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color='time_categorical',
        figsize=(7, 5),
        density=3,
        title='Interaction Force Stream Plot',
        legend_loc='right',
        palette='plasma',
        save='Interaction_Force_Stream_Plot.svg',
    )
    print(f"Interaction force stream plot saved to: {output_path}/Interaction_Force_Stream_Plot.svg")
    return adata
#%%
import anndata
import scvelo as scv
import scanpy as sc
import numpy as np
import pandas as pd
import torch

def plot_score_stream(adata,model,output_path, dim_reduction='none', device='cuda'):
    """
    Plot score function diffusion velocity field streamlines
    Coordinates use first two dimensions of adata.obsm['X_latent'] or dimensionality reduction results
    Velocity is calculated from the gradient of the score function (log density)
    """
    # 1. Check if necessary data exists
    required_keys = ['X_latent', 'time_point_processed']
    for key in required_keys:
        if (key not in adata.obsm) and (key not in adata.obs):
            raise ValueError(f"Required data not found: {key}, please ensure the model has correctly computed relevant results")

    # 2. Prepare input data (latent space data and time)
    device = torch.device(device)
    all_data = torch.tensor(adata.obsm['X_latent'], dtype=torch.float32, device=device)
    all_data.requires_grad_(True)  # Enable gradient tracking (critical modification)
    all_times = torch.tensor(adata.obs['time_point_processed'].values, dtype=torch.float32, device=device).unsqueeze(1)

    # 3. Calculate score function (log density) and gradients (velocity vectors)
    # Calculate log density values directly from the model, ensuring complete computation graph
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
        from umap import UMAP          
        reducer = UMAP().fit(adata.obsm['X_latent'])
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
    scv.settings.set_figure_params(dpi=100, figsize=(7, 5))

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

def plot_combined_velocity_stream(adata,model,output_path, dim_reduction='none', device='cuda'):
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
        adata = compute_velocity(adata, model,device)
    vel = adata.layers['velocity_latent'].copy()  # Model velocity
    norms = np.linalg.norm(vel, axis=1, keepdims=True)
    vel_model = np.divide(vel, norms, out=np.zeros_like(vel), where=norms!=0) * 5
    # 1.2 Calculate score gradient velocity
    device = torch.device(device)
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
    scv.settings.set_figure_params(dpi=100, figsize=(7, 5))

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
    print(f"All velocity stream plot saved to: {output_path}/All_Velocity_Stream.svg")
    return adata
    #%%
import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_interaction_potential(
    model,
    d=200,
    num_points=100,
    output_path=None,
    device="cuda"
):
    """
    Visualize interaction potential energy (input is distance d, output is energy value)
    
    Parameters:
        model: Trained model containing interaction network
        d: Range parameter for distance (d_min = -d, d_max = d)
        num_points: Number of sampling points
        output_path: Path to save the plot (do not save if None)
        device: Computing device
    """
    d_min = -d
    d_max = d
    # Load model
    model.to(device)
    model.eval()
    # Initialize ODE function (used for energy calculation)
    
    # Generate distance sequence
    d_values = np.linspace(d_min, d_max, num_points)
    d_tensor = torch.tensor(d_values, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension to match network input
    # Calculate energy (use no_grad to speed up and avoid gradient computation)
    with torch.no_grad():
        energies = model.interaction_net(d_tensor)
        # Critical fix: Move CUDA tensor to CPU and convert to NumPy array
        energies = energies.cpu().numpy().flatten()  # Flatten to 1D array to match d_values shape
    print(energies)

    # Visualization
    plt.figure(figsize=(4.8, 3))
    plt.plot(d_values, energies, 'b-', linewidth=2)
    plt.xlabel('Distance (d)')
    plt.ylabel('Interaction Potential Energy')
    plt.title('Interaction Potential vs Distance')
    plt.grid(True, alpha=0.3)
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Interaction potential plot saved to: {output_path}")
    else:
        plt.show()

def plot_interaction_potential_epoch(
    model,
    d=200,
    num_points=100,
    output_path=None,
    device="cuda"
):
    """
    Visualize interaction potential energy (input is distance d, output is energy value)
    
    Parameters:
        model: Trained model containing interaction network
        d: Range parameter for distance (d_min = -d, d_max = d)
        num_points: Number of sampling points
        output_path: Path to save the plot (do not save if None)
        device: Computing device
    """
    d_min = -d
    d_max = d
    # Load model
    model.to(device)
    model.eval()
    # Initialize ODE function (used for energy calculation)
    
    # Generate distance sequence
    d_values = np.linspace(d_min, d_max, num_points)
    d_tensor = torch.tensor(d_values, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension to match network input
    # Calculate energy (use no_grad to speed up and avoid gradient computation)

    with torch.no_grad():
        energies = model.interaction_net(d_tensor)
        # Critical fix: Move CUDA tensor to CPU and convert to NumPy array
        energies = energies.cpu().numpy().flatten()  # Flatten to 1D array to match d_values shape
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, energies, 'b-', linewidth=2)
    plt.xlabel('Distance (d)')
    plt.ylabel('Interaction Potential Energy')
    plt.title('Interaction Potential vs Distance')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        last_slash_index = output_path.rfind('/')
        new_path = output_path[:last_slash_index]
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print(f"Directory created: {new_path}")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Interaction potential plot saved to: {output_path}")
    else:
        plt.show()

def plot_interaction_potential_epoch(
    model,
    d=200,
    num_points=100,
    output_path=None,
    device="cuda"
):
    """
    Visualize interaction potential energy (input is distance d, output is energy value)
    
    Parameters:
        model: Trained model containing interaction network
        d: Range parameter for distance (d_min = -d, d_max = d)
        num_points: Number of sampling points
        output_path: Path to save the plot (do not save if None)
        device: Computing device
    """
    d_min = -d
    d_max = d
    # Load model
    model.to(device)
    model.eval()
    # Initialize ODE function (used for energy calculation)
    
    # Generate distance sequence
    d_values = np.linspace(d_min, d_max, num_points)
    d_tensor = torch.tensor(d_values, dtype=torch.float32, device=device).unsqueeze(1)  # Add dimension to match network input
    # Calculate energy (use no_grad to speed up and avoid gradient computation)
    with torch.no_grad():
        energies = model.interaction_net(d_tensor)
        # Critical fix: Move CUDA tensor to CPU and convert to NumPy array
        energies = energies.cpu().numpy().flatten()  # Flatten to 1D array to match d_values shape
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, energies, 'b-', linewidth=2)
    plt.xlabel('Distance (d)')
    plt.ylabel('Interaction Potential Energy')
    plt.title('Interaction Potential vs Distance')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        last_slash_index = output_path.rfind('/')
        new_path = output_path[:last_slash_index]
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            print(f"Directory created: {new_path}")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Interaction potential plot saved to: {output_path}")
    else:
        plt.show()
#%%
# Fit potential on 2D UMAP to create landscape
import torch.nn as nn
import torch.optim as optim
import torch

def plot_landscape(adata,model,output_path=None,dim_reduction="none",device="cuda"):

    # 1. Ensure velocity has been calculated
    if 'velocity_latent' not in adata.layers:
        adata = compute_velocity(adata, model,device)

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

    if 'velocity_latent' not in adata.layers:
        adata = compute_velocity(adata, model,device)
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


    X = torch.tensor(adata.obsm['X_umap'], dtype=torch.float32)
    V = torch.tensor(adata.obsm['velocity_umap'], dtype=torch.float32)

    class PotentialNet(nn.Module):
        def __init__(self, in_dim=2, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    potential_net = PotentialNet(in_dim=X.shape[1])
    potential_net = potential_net.cuda()
    X = X.cuda()
    V = V.cuda()

    optimizer = optim.Adam(potential_net.parameters(), lr=1e-3)
    n_epochs = 2000

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        X.requires_grad = True
        phi = potential_net(X)
        grad_phi = torch.autograd.grad(
            phi, X, 
            grad_outputs=torch.ones_like(phi), 
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True
        )[0]  # (N, 2)
        # Velocity is the negative gradient of potential
        pred_V = -grad_phi
        loss = ((pred_V - V)**2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, PotentialNet_loss: {loss.item():.6f}")

    # Calculating Potential for original cells
    umap_coords = adata.obsm['X_umap']
    with torch.no_grad():
        input_tensor = torch.from_numpy(umap_coords).float().to(device)
        original_potentials = potential_net(input_tensor).cpu().numpy().flatten()

    # Plot potential
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, 4))

    scatter = sns.scatterplot(
        x=umap_coords[::1, 0],
        y=umap_coords[::1, 1],
        hue=original_potentials[::1],
        palette='RdYlBu_r', 
        s=5,              
        alpha=0.8,           
        edgecolor='none',     
        ax=ax,
        legend=False   
    )


    norm = plt.Normalize(original_potentials.min(), original_potentials.max())
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Potential', fontsize=12)


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_aspect('equal', adjustable='box')
    plt.savefig(output_path+'/Potential_landscape.png', dpi=100, bbox_inches='tight')
    plt.grid(False)

    plt.tight_layout()
    plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import cellrank as cr
import os


def analyze_terminal_states(adata, classified_type="cell_type", terminal_states=["type1", "type2"], output_path=None):
    """
    Analyze and plot terminal states, and compute fate probabilities.

    Parameters:
    adata: AnnData object containing single-cell data and velocity information.
    classified_type: String indicating the column in adata.obs that contains cell type classifications.
    terminal_states: List of strings representing the terminal states to be analyzed.
    output_path: String specifying the folder path where results will be saved.
    """

    # Check if classified_type exists in adata.obs
    if classified_type not in adata.obs.columns:
        raise ValueError(f"Error: '{classified_type}' does not exist in adata.obs. Please check the column name.")

    # Check if all terminal_states exist in classified_type
    missing_states = [state for state in terminal_states if state not in adata.obs[classified_type].unique()]
    if missing_states:
        raise ValueError(f"Error: The following terminal states do not exist in '{classified_type}': {missing_states}")

    # Ensure output folder exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if 'neighbors' not in adata.uns:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, use_rep='X_latent')
    # Create VelocityKernel object and compute transition matrix
    vk = cr.kernels.VelocityKernel(adata)
    vk.compute_transition_matrix()
    transition_matrix = vk.transition_matrix
    if not np.allclose(np.sum(transition_matrix, axis=1), 1, atol=1e-3):
        raise ValueError("Transition matrix rows do not sum to 1.")
    # Create GPCCA object
    g = cr.estimators.GPCCA(vk)

    # Manually define terminal states
    manual_terminal_states = {
        state: adata.obs_names[adata.obs[classified_type] == state].tolist()
        for state in terminal_states
    }

    # Set terminal states
    g.set_terminal_states(manual_terminal_states)

    # Plot terminal states
    g.plot_macrostates(which="terminal", mode = 'embedding', legend_loc="right margin",)

    print(f"Terminal states analysis is saved to {output_path}")
    g.compute_fate_probabilities(tol=1e-5, preconditioner="ilu")
    g.plot_fate_probabilities(mode="embedding", title='All fate probabilities', save=output_path + '/all_fate_probabilities_plot.svg')
#%%
from CytoBridge.tl.analysis import train_mlp_classifier,MLPClassifier
from anndata import AnnData
from typing import  Optional
import joblib

def visualize_trajectory_classification(
    sde_point: np.ndarray,
    predicted_labels_list: list,
    output_path: str,
    adata: AnnData,
    time_points: Optional[np.ndarray] = None
):
    """
    Visualize cell type distribution at different time points
    
    Parameters:
        sde_point: SDE-generated trajectory points (n_time_points, n_cells, latent_dim)
        predicted_labels_list: List of predicted labels for each time point
        output_path: Path to save visualization results
        adata: Original data used to obtain time point information
        time_points: Time point array (optional)
    """
    os.makedirs(output_path, exist_ok=True)
    cmap = plt.cm.get_cmap('tab20')
    
    # Get all unique labels and map to colors
    all_labels = np.unique(np.concatenate(predicted_labels_list))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    # Get time points (extract from adata if not provided)
    if time_points is None:
        time_points = sorted(adata.obs['time_point_processed'].unique())
    
    # Plot for each time point
    for i in range(len(time_points)):
        t = time_points[i]
        # Extract SDE data for current time point
        data_t = sde_point[i]
        # Extract predicted labels
        labels_t = predicted_labels_list[i]
        
        # Convert labels to indices (for color mapping)
        label_indices = np.array([label_to_idx[label] for label in labels_t])
        colors = cmap(label_indices % cmap.N)
        
        # Plot (assuming data is already dimensionality-reduced, using first two dimensions)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(data_t[:, 0], data_t[:, 1], c=colors, alpha=0.8, s=30)
        ax.set_title(f'Cell Type Distribution at Time {t}')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        
        # Add legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i % cmap.N), markersize=10) 
                   for i in range(len(all_labels))]
        ax.legend(handles, all_labels, title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'trajectory_time_{t}.png'), dpi=100)
        plt.show()
        plt.close()

def process_sde_classification(
    adata: AnnData,
    model: nn.Module,
    classifyed_type: str = 'cluster',
    sde_data_path: str = './sde_data',
    output_path: str = './classification_results',
    sigma: float = 0.05,
    n_time_steps: int = 10,
    sample_traj_num: int = 1000,
    train_mlp_classifier_epoches=400,
    init_time: Optional[int] = None,
    device: str = 'cuda',
    dim_reduction="none",
    hidden_size: int = 128  # Add hidden_size parameter to ensure consistency during loading and training
):
    """
    Complete workflow: Train classifier → Process SDE data → Classification prediction → Visualization
    
    Parameters:
        adata: AnnData object containing cell data
        model: Pretrained SDE model (used for generating trajectories)
        classifyed_type: Key of classification target in obs (default 'cluster')
        sde_data_path: Path to save SDE trajectory data
        output_path: Path to save results
        sigma: SDE noise parameter
        n_time_steps: Number of time steps
        sample_traj_num: Number of trajectories to sample
        init_time: Initial time point (uses the earliest time point in adata by default)
        device: Computing device
        hidden_size: MLP hidden layer dimension, used to maintain consistency when loading the model
    """
    model_path = os.path.join(output_path, 'mlp_classifier.pth')
    encoder_path = os.path.join(output_path, 'label_encoder.pkl')
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        print("Loading existing MLP classifier...")
        label_encoder = joblib.load(encoder_path)
        input_size = adata.X.shape[1]
        num_classes = len(label_encoder.classes_)
        mlp_model = MLPClassifier(input_size, hidden_size, num_classes).to(device)
        mlp_model.load_state_dict(torch.load(model_path, map_location=device))
        mlp_model.eval()  
    else:
        print("Training MLP classifier...")
        mlp_model, label_encoder = train_mlp_classifier(
            adata,
            classifyed_type=classifyed_type,
            hidden_size=hidden_size,  
            device=device,
            train_mlp_classifier_epoches=train_mlp_classifier_epoches
        )
        # Save classifier
        os.makedirs(output_path, exist_ok=True)
        torch.save(mlp_model.state_dict(), model_path)
        joblib.dump(label_encoder, encoder_path)
    
    # 2. Process SDE data
    sde_point_path = os.path.join(sde_data_path, 'sde_point.npy')
    if not os.path.exists(sde_point_path):
        print("Generating SDE trajectories...")
        
        if init_time is None:
            init_time = sorted(adata.obs['time_point_processed'].unique())[0]
        # Generate SDE trajectories
        generate_sde_trajectories(
            adata=adata,
            exp_dir=sde_data_path,
            device=device,
            sigma=sigma,
            n_time_steps=n_time_steps,
            sample_traj_num=sample_traj_num,
            init_time=init_time
        )
    
    # Load SDE data
    print("Loading SDE data...")
    sde_point = np.load(sde_point_path)
    time_points = sorted(adata.obs['time_point_processed'].unique())
    assert len(sde_point) == len(time_points), "Number of time points in SDE data does not match adata"
    

    # 3. MLP classification prediction
    print("Predicting cell types for SDE trajectories...")
    mlp_model.eval()
    predicted_labels_list = []
    with torch.no_grad():
        for i in range(len(time_points)):
            # Get SDE data at current time point
            data_t = sde_point[i]
            data_t_tensor = torch.tensor(data_t, dtype=torch.float32).to(device)
            
            # Make predictions
            outputs = mlp_model(data_t_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.cpu().numpy())
            predicted_labels_list.append(predicted_labels)

    if (sde_point.shape[-1]==2):
        point_2d = sde_point[..., :2]  
    elif dim_reduction == 'pca':
        mean_ = adata.uns['pca']['mean'] if 'mean' in adata.uns['pca'] else 0.
        comps = adata.uns['pca']['components_'][:, :2]
        point_2d = np.dot(sde_point - mean_, comps)  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)
    elif dim_reduction == 'umap':
        from umap import UMAP       
        reducer = UMAP().fit(adata.obsm['X_latent'])
        def flatten_and_2d(model, x):
            *lead, F = x.shape               
            x_2d = x.reshape(-1, F)         
            y_2d = model.transform(x_2d)    
            return y_2d.reshape(*lead, 2)   
        point_2d = flatten_and_2d(reducer, sde_point) # (n_time, n_traj, 2)
        
    else:  
        point_2d = sde_point[..., :2]  # Shape: [n_time_points, sample_traj_num, 2] (reduced when consistent)


    print("Visualizing results...")
    visualize_trajectory_classification(
        sde_point=point_2d,
        predicted_labels_list=predicted_labels_list,
        output_path=output_path,
        adata=adata,
        time_points=time_points
    )
    print(f"Results saved to {output_path}")