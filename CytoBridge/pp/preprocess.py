import scanpy as sc
from anndata import AnnData
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple


def preprocess(
    adata: AnnData,
    time_key: str,
    n_top_genes: int = 2000,
    dim_reduction: str = 'pca',
    n_pcs: int = 50,
    time_mapping: Optional[Dict[str, float]] = None,
    normalization: bool = True,
    log1p: bool = True,
    select_hvg: bool = True,
) -> Tuple[AnnData, Optional[AnnData]]:
    """
    Preprocess step for dynamical optimal transport analysis.

    This function implements a standard preprocessing workflow. Key steps include:
    1. Mapping categorical time points to a numerical scale.
    2. Normalizing and log-transforming the data.
    3. Selecting highly variable genes (HVGs).
    4. Scaling the data and performing PCA.

    Parameters
    ----------
    adata
        The annotated data matrix.
    time_key
        The key in `adata.obs` that specifies the time point of each cell.
    n_top_genes
        Number of highly variable genes to select.
    dim_reduction
        The dimension reduction method to use.
    n_pcs
        Number of principal components to compute.
    time_mapping
        A dictionary to map non-numeric time points to numeric values.
        Example: {'Day0': 0, 'Day2': 2.0, 'Day7': 7.0}.
        If None (default), time points are automatically sorted and mapped to
        0, 1, 2, ...
    normalization
        If True, normalize the data.
    log1p
        If True, apply log1p transformation to the data.
    select_hvg
        If True, select highly variable genes.
    Returns
    -------
    Tuple containing:
        - processed_adata: Original processed `AnnData` object (unchanged structure)
        - n_pc_adata: New `AnnData` object with reduced dimension as `X` (only created when 
          dim_reduction is 'pca' or 'umap'). Its structure:
            - X: Reduced dimension data (n_samples × n_dimensions, from X_latent)
            - obs: Complete original obs information
            - var: New var information for reduced dimensions (dimension names and properties)
            - uns['original_gene_info']: Stores original gene-related information including:
                - var: Original var DataFrame
                - var_names: Original gene names
                - X_shape: Original X shape (n_samples × n_genes)
                - hvg_mask: Highly variable gene mask (if HVG selection was applied)
    """
    # --- Input Validation and Setup ---
    if time_key not in adata.obs.keys():
        raise KeyError(
            f"The specified time_key '{time_key}' was not found in adata.obs. "
            f"Available keys are: {list(adata.obs.keys())}"
        )
    print(f"Using '{time_key}' as the time point identifier.")

    # --- Time Point Mapping Logic ---
    unique_times = adata.obs[time_key].unique()

    # The processed time key is current hard-coded for convenience
    time_key_added = 'time_point_processed'
    if time_mapping:
        print(f"Using user-provided time mapping.")
        # Check if all time points in data are present in the mapping
        missing_keys = [t for t in unique_times if t not in time_mapping]
        if missing_keys:
            raise ValueError(
                f"The following time points in adata.obs['{time_key}'] are "
                f"not present in the provided time_mapping: {missing_keys}"
            )
        # Apply the user-defined mapping
        adata.obs[time_key_added] = adata.obs[time_key].map(time_mapping).astype(float)

    else:
        print("No time mapping provided. Generating automatic mapping.")
        # Automatically sort unique time points and map them to integers
        sorted_times = sorted(unique_times)
        auto_mapping = {time_point: i for i, time_point in enumerate(sorted_times)}
        print(f"Automatically generated time mapping: {auto_mapping}")
        # Apply the automatic mapping
        adata.obs[time_key_added] = adata.obs[time_key].map(auto_mapping).astype(float)

    print(f"Numerical time points stored in `adata.obs['{time_key_added}']`.")

    # --- Standard Preprocessing Steps ---
    if normalization:
        print("Normalizing total counts and applying log1p transformation.")
        sc.pp.normalize_total(adata, target_sum=1e4)

    if log1p:
        sc.pp.log1p(adata)

    # Record the HVG mask (for later storage in original_gene_info)
    hvg_mask = None
    if select_hvg:
        print(f"Selecting top {n_top_genes} highly variable genes.")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
        )
        hvg_mask = adata.var.highly_variable.copy()
        adata = adata[:, hvg_mask]

    # --- Dimension Reduction ---
    # ---------- : PCA | UMAP | none ----------
    n_pc_adata = None  # Initialize n_pc_adata as None
    if dim_reduction.lower() == 'pca':
        sc.pp.pca(adata, n_comps=n_pcs, svd_solver='arpack')
        adata.obsm['X_latent'] = adata.obsm['X_pca']
        # Create n_pc_adata
        n_dim = n_pcs

        dim_names = [f'PC{i+1}' for i in range(n_dim)]
        
        # Construct the new var (to match the reduced dimensions)
        new_var = pd.DataFrame(
            index=dim_names,
            data={

                'dimension_type': 'PCA',
                'explained_variance': adata.uns['pca']['variance_ratio'][:n_dim] if 'pca' in adata.uns else np.nan,
                'cumulative_variance': np.cumsum(adata.uns['pca']['variance_ratio'][:n_dim]) if 'pca' in adata.uns else np.nan
            }
        )

    elif dim_reduction.lower() == 'umap':
        sc.pp.pca(adata, n_comps=n_pcs, svd_solver='arpack')
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.umap(adata)
        adata.obsm['X_latent'] = adata.obsm['X_umap']
        
        # Create n_pc_adata
        n_dim = adata.obsm['X_umap'].shape[1]  # Usually 2
        dim_names = [f'UMAP{i+1}' for i in range(n_dim)]
        
        # Construct the new var (to match the reduced dimensions)
        new_var = pd.DataFrame(
            index=dim_names,
            data={
                'dimension_type': 'UMAP',
                'n_neighbors_used': adata.uns['neighbors']['params']['n_neighbors'] if 'neighbors' in adata.uns else np.nan,
                'min_dist_used': adata.uns['umap']['params']['min_dist'] if 'umap' in adata.uns else np.nan
            }
        )

    elif dim_reduction.lower() == 'none' or dim_reduction is None:
        adata.obsm['X_latent'] = adata.X
        n_pc_adata = adata
        print("Dimension reduction set to 'none', not creating n_pc_adata.")

    else:
        raise ValueError(f"Invalid dimension reduction method: {dim_reduction}")

    # If PCA or UMAP, complete the creation of n_pc_adata
    if dim_reduction.lower() in ['pca', 'umap']:
        # Extract the reduced data as the new X
        reduced_X = adata.obsm['X_latent'].copy()
        
        # Create the core structure of n_pc_adata
        n_pc_adata = AnnData(
            X=reduced_X,
            obs=adata.obs.copy(deep=True),  # Fully retain the original obs
            var=new_var,
            uns=adata.uns.copy()  # Copy the original uns, to be supplemented with original_gene_info later
        )
        
        # Copy the original obsm (excluding X_latent to avoid redundancy)
        for key, value in adata.obsm.items():
                n_pc_adata.obsm[key] = value.copy()
        
        # Copy the original obsp
        if adata.obsp is not None and len(adata.obsp) > 0:
            n_pc_adata.obsp = adata.obsp.copy()
        
        # Store the original gene-related information in uns['original_gene_info']
        original_gene_info = {
            'var': adata.var.copy(deep=True),  # Original var (after HVG selection)
            'var_names': adata.var_names.tolist(),  # Original gene name list
            'X_shape': np.array(adata.X.shape),  # Original X shape (n_samples × n_genes)
            'select_hvg': select_hvg,  # Whether HVG selection was performed
        }

        # If hvg_mask exists, add it to the dictionary
        if hvg_mask is not None:
            original_gene_info['hvg_mask'] = hvg_mask.values

        # If n_top_genes exists, add it to the dictionary
        if n_top_genes is not None:
            original_gene_info['n_top_genes'] = n_top_genes

        n_pc_adata.uns['original_gene_info'] = original_gene_info
        
        print(f"Created n_pc_adata with reduced dimension: {reduced_X.shape[0]} samples × {reduced_X.shape[1]} dimensions")
        print(f"Original gene information stored in n_pc_adata.uns['original_gene_info']")

    print("Preprocessing recipe finished.")
    return n_pc_adata