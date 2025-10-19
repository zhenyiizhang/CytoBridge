
import scanpy as sc
from anndata import AnnData
import logging
from typing import Optional, Dict


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
) -> Optional[AnnData]:
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
    A new processed `AnnData` object.
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

    if select_hvg:
        print(f"Selecting top {n_top_genes} highly variable genes.")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
        )
        adata = adata[:, adata.var.highly_variable]

    # --- Dimension Reduction ---
    # ---------- ：PCA | UMAP | none ----------
    if dim_reduction.lower() == 'pca':
        # 1. 如果还没算过 PCA，就现场算
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=n_pcs, svd_solver='arpack')
        # 2. 把 PCA 结果当作“潜空间”
        adata.obsm['X_latent'] = adata.obsm['X_pca']

    elif dim_reduction.lower() == 'umap':
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=n_pcs, svd_solver='arpack')
        sc.pp.neighbors(adata, n_pcs=n_pcs)
        sc.tl.umap(adata)
        adata.obsm['X_latent'] = adata.obsm['X_umap']

    elif dim_reduction.lower() == 'none' or dim_reduction is None:
        # 直接用原始表达矩阵
        adata.obsm['X_latent'] = adata.X

    else:
        raise ValueError(f"Invalid dimension reduction method: {dim_reduction}")

    print("Preprocessing recipe finished.")

    return adata

