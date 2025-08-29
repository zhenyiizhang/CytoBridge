import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

def plot_growth(adata, dim_reduction = 'umap', save_path = None):
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

    # Get growth values
    g_values = adata.obsm['growth_rate']

    # Calculate 95th percentile of g_values
    vmax_value = np.percentile(g_values, 95)

    # Initialize color mapper with clipping
    norm = plt.Normalize(vmin=0, vmax=vmax_value, clip=True)

    # Get colors
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

    # Save figure
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

    plt.show()
