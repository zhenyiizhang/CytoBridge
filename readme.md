<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/zhenyiizhang/CytoBridge_test/">
    <img src="figures/logo.png" alt="Logo" height="150" style="margin-bottom: 0px;">
  </a>
  
  <h3 align="center">CytoBridge: A Toolkit for Spatiotemporal Dynamical Generative Modeling</h3>

</div>

**CytoBridge** is a Python package designed to provide a unified and extensible framework for modeling cellular dynamics using Spatiotemporal Dynamical Generative Models. Our goal is to connect cutting-edge theoretical frameworks with practical, easy-to-use tools.

The core of CytoBridge is to model cellular processes by learning a dynamical model that accounts for various biological phenomena using time-series transcriptomics data (including both scRNA-seq and spatial data):

  * **Velocity**: The instantaneous direction of cell state transitions. (Corresponds to **Dynamical OT**)
  * **Growth**: The proliferation and apoptosis rates of cell populations. (Corresponds to **Unbalanced Optimal Transport, UOT**)
  * **Score**: (Planned) A diffusion or noise term, capturing the stochasticity in cell state. (Corresponds to **Regularized Unbalanced OT**)
  * **Interaction**: (Planned) A term to model cell-cell communication and its effect on dynamics (Corresponds to **Unbalanced Mean Field Schrödinger Bridge**).

The package is built to be modular, allowing users to easily combine these components to replicate existing models or create novel ones.

## Current Status

CytoBridge is currently under active development. The foundational framework is in place, and we have implemented the following models:

  * **Dynamical OT** (velocity)
  * **Unbalanced Dynamical OT** (velocity + growth)

## Roadmap

We are continuously working to expand the capabilities of CytoBridge. Our development plan is as follows:

  - [ ] **Phase 1: Stochastic Dynamics & RUOT**

      - [ ] Implementation of the `score` component to model stochasticity.
      - [ ] Support for training methods based on the **Regularized Unbalanced Optimal Transport (RUOT)** frameworks.
      - [ ] Integration of simulation-free training methods (e.g., Conditional Flow Matching, Velocity-growth Flow Matching).
      - [ ] Basic plotting functions and downstream analysis.

  - [ ] **Phase 2: Advanced Modeling & Downstream Analysis**

      - [ ] Implementation of the `interaction` component for modeling cell-cell communication.
      - [ ] Advanced plotting functions and downstream analysis.


  - [ ] **Phase 3: Spatiotemporal Dynamics**

      - [ ] Support for time serise spatial transcriptomics data.
      - [ ] Advanced plotting functions and downstream analysis.
## Installation

Currently, CytoBridge is in an early development stage. You can download it directly from this repository for testing and contribution:

```bash
git clone https://github.com/zhenyiizhang/CytoBridge.git
```

*(A PyPI release is planned for the future.)*

## Basic Usage

You can create a new conda environment (CytoBridge) and install required packages using

```vim
conda create -n CytoBridge python=3.10 ipykernel -y
conda activate CytoBridge
cd path_to_CytoBridge
pip install -r requirements.txt
```

Here is a simple example of how to use CytoBridge, from preprocessing data to training a model and saving the results.

```python
import scanpy as sc
import cytobridge as cb
import anndata as ad

# 1. Load your data and preprocess
# This assumes your adata object has an obs column for timepoints
cb.pp.preprocess(adata, time_key = 'Time point', dim_reduction = 'PCA', normalization = True, log1p = True, select_hvg = True)

# 2. Train a model using a built-in configuration
# The 'dynamical_ot' config uses the currently implemented 'velocity' components.
# The 'unbalanced_ot' config uses the currently implemented 'velocity' components.
# The `fit` function runs the entire training plan defined in the config.
# The trained model and results are automatically saved to adata.uns['dynamic_model'], adata.obsm['velocity_latent'], adata.obsm['growth_rate']
cb.tl.fit(adata, config='dynamical_ot', device='cuda')

# 3. Save the AnnData object with the model for later use
adata.write_h5ad("results_with_model.h5ad")
```

## References

1. Zhenyi Zhang, Tiejun Li, and Peijie Zhou. “Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport”. In: The Thirteenth International Conference on Learning Representations, 2025.
2. Zhenyi Zhang, Yuhao Sun, Qiangwei Peng, Tiejun Li, and Peijie Zhou. “Integrating Dynamical Systems Modeling with Spatiotemporal scRNA-Seq Data Analysis”. In: Entropy 27.5, 2025b. ISSN: 1099-4300.
3. Zhenyi Zhang, Zihan Wang, Yuhao Sun, Tiejun Li, and Peijie Zhou. “Modeling Cell Dynamics and Interac- tions with Unbalanced Mean Field Schrödinger Bridge”. In: arXiv preprint arXiv:2505.11197, 2025c.
4. Yuhao Sun, Zhenyi Zhang, Zihan Wang, Tiejun Li, and Peijie Zhou. “Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action”. In: arXiv preprint arXiv:2505.11823, 2025.
5. Dongyi Wang, Yuanwei Jiang, Zhenyi Zhang, Xiang Gu, Peijie Zhou, and Jian Sun. “Joint Velocity-Growth Flow Matching for Single-Cell Dynamics Modeling”. In: arXiv preprint arXiv:2505.13413, 2025.
6. Qiangwei Peng, Peijie Zhou, and Tiejun Li. “stVCR: Reconstructing spatio-temporal dynamics of cell development using optimal transport”. In: bioRxiv, 2024, pp. 2024–06.
7. https://github.com/zhenyiizhang/DeepRUOTv2