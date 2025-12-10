# CytoBridge Deep Dive: A Comprehensive Tutorial 

Welcome to the detailed tutorial on using CytoBridge for single-cell spatiotemporal dynamical modeling. This guide is designed to provide an in-depth walkthrough of the entire process, from data preparation and preprocessing to PCA extraction and model training. Additionally, it offers practical insights into selecting the most suitable models and components based on specific biological questions and computational limitations.

## Table of Contents

- [1. Data Requirements](#data-requirements)
- [2. Quick Start](#quick-start)
- [3. Full Workflow Walkthrough](#full-workflow-walkthrough)
  - [3.1 Preprocessing](#preprocessing)
  - [3.2 PCA Extraction](#pca-extraction)
  - [3.3 Model Training](#model-training)
- [4. Model Selection Guide](#model-selection-guide)
  - [4.1 Component Selection (Biology-Driven)](#component-selection-biology-driven)
  - [4.2 Training Mode Selection (Computation-Driven)](#training-mode-selection-computation-driven)
  - [4.3 Model Selection Cheat Sheet](#model-selection-cheat-sheet)
- [5. Loss Functions](#loss-functions)
  - [5.1 OT Loss (Optimal Transport Loss)](#ot-loss-optimal-transport-loss)
  - [5.2 Mass Loss](#mass-loss)
  - [5.3 Score Matching Loss](#score-matching-loss)
  - [5.4 Density Loss](#density-loss)
  - [5.5 PINN Loss (Physics-Informed Neural Network Loss)](#pinn-loss)
  - [5.6 Loss Selection Cheat Sheet](#loss-selection-cheat-sheet)

## 1. Data Requirements

CytoBridge is designed for **single-cell transcriptomic data (scRNA-seq/spatial transcriptomics)** with temporal annotations. Below are the supported formats and critical requirements:



### Supported Input Formats

| Format                           | Use Case                                                     | Required Structure                                           |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **AnnData (h5ad)** (Recommended) | Standard scRNA-seq/spatial data (compatible with Scanpy/Seurat) | - `adata.X`: Gene expression matrix (cells × genes, raw counts/normalized values)- `adata.obs`: Metadata with a **time column** (e.g., `Time_point`, `Day`)- Optional: Cell type annotations |
| **CSV**                          | Lightweight datasets                                         | - Rows = cells, Columns = gene counts + 1 time column (e.g., `samples`, `Time`)- No missing values in the time column |



### Critical Annotations

* **Time Key**: A column in `adata.obs` (or CSV) that labels each cell with its experimental time point (e.g., categorical: `Day0`/`Day2`; numerical: `0`/`2.0`).

  *This is mandatory for modeling temporal dynamics.*

* **Raw Counts (Preferred)**: CytoBridge’s preprocessing pipeline is optimized for raw scRNA-seq counts (normalization/log1p steps are configurable).



## 2. Quick Start

### Get up and running in 5 minutes with a minimal example (AnnData input):

```python
import scanpy as sc
import CytoBridge as cb

# 1. Load data
adata = sc.read_h5ad("raw_scrna_data.h5ad")

# 2. Preprocess (normalize + HVG + PCA)
time_mapping = {"Day0": 0.0, "Day2": 2.0, "Day5": 5.0}  # Map time points to numbers
n_pc_adata = cb.pp.preprocess(
    adata=adata,
    time_key="Time_point",
    time_mapping=time_mapping,
    normalization=True,
    log1p=True,
    select_hvg=True,
    dim_reduction="pca",
    n_pcs=50
)

# 3. Train model (Velocity + Growth + Score)
cb.tl.fit(
    adata=n_pc_adata,
    config="ruot",          # Velocity + Growth + Score
    device="cuda",          # Use "cpu" if no GPU
)

# 4. Save results
n_pc_adata.write_h5ad("trained_model.h5ad")
```



## 3. Full Workflow Walkthrough

### 3.1 Preprocessing

Preprocessing reduces noise and extracts biologically meaningful features for dynamical modeling. The `cb.pp.preprocess` function encapsulates a standardized workflow—tune parameters based on your data type.



#### Key Preprocessing Steps & Parameters

| Step                     | Purpose                                             | Parameter                         | When to Use                                                  |
| ------------------------ | --------------------------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| **Normalization**        | Scale counts to fix sequencing depth differences    | `normalization=True`              | Enable for raw counts; disable for pre-normalized data       |
| **Log1p Transformation** | Stabilize variance in gene expression               | `log1p=True`                      | Enable for raw counts; disable for already log-transformed data |
| **HVG Selection**        | Focus on biologically variable genes (reduce noise) | `select_hvg=Truen_top_genes=2000` | Enable for scRNA-seq; disable for sparse spatial transcriptomics |
| **Time Mapping**         | Convert categorical time labels to numbers          | `time_mapping={"Day0":0.0}`       | Always define explicitly (avoids arbitrary time ordering)    |



#### Full Preprocessing Code

```python
import scanpy as sc

import CytoBridge as cb

# Load data

adata = sc.read_h5ad("raw_scrna_data.h5ad")

# Define time mapping (critical for temporal dynamics)

time_mapping = {"Day0": 0.0, "Day2": 2.0, "Day5": 5.0}

# Run preprocessing

n_pc_adata = cb.pp.preprocess(

   adata=adata,

   time_key="Time_point",          # Column with time labels

   time_mapping=time_mapping,      # Explicit time mapping (avoid auto-mapping)

   normalization=True,             # Raw counts → True; normalized → False

   log1p=True,                     # Raw counts → True; log-transformed → False

   select_hvg=True,                # scRNA-seq → True; spatial → False

   n_top_genes=2000,               # Adjust for data size (2000 for small datasets)

   dim_reduction="pca",            # Use PCA for modeling 
    
   n_pcs=50                        # 30-50 for scRNA-seq; 20-30 for spatial data

)

# Verify preprocessing output

print(f"Preprocessed PCA data shape: {n_pc_adata.shape}")  # (n_cells, n_pcs)

print(f"Latent space: {n_pc_adata.obsm['X_latent'].shape}")  # latent results stored here 
# when you choose dim_reduction="none",it will train original data. (latent) 
# when you choose dim_reduction="pca",it will train n_pc_adata.	(latent)	

print(f"Processed time points: {n_pc_adata.obs['time_point_processed'].unique()}")
```



### 3.2 PCA Extraction

PCA is the **gold standard** for linear dimension reduction in CytoBridge—it preserves global dynamical structure (unlike UMAP/t-SNE, which are for visualization only).

#### Why PCA?

* Retains linear relationships between cell states (critical for modeling velocity/growth).

* Reduces computational cost of training (50 PCs vs. 20k genes).

* Minimizes noise while preserving biological signal.

  

### 3.3 Model Training

Use the preprocessed PCA data (`n_pc_adata`) to train a CytoBridge model. Choose a `config` (component combination) and `mode` (training method) based on your biological question.

#### Core Training Parameters

| Parameter | Description                                                  | Options                                                      |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `config`  | Defines model components (velocity/growth/score/interaction) | `dynamical_ot` (velocity only);`unbalanced_ot` (velocity + growth); `ruot` (velocity + growth + score); `crufm` (velocity + growth + score); `cytobridge` (all components) |
| `device`  | Compute device                                               | `cuda` (GPU, fast)`cpu` (slow, no GPU)                       |
| `mode`    | Training algorithm                                           | `flow_matching` (fast)`neural_ode` (comprehensive)           |

#### Training Examples

##### Example 1: Stem Cell Differentiation (Velocity + Growth + Score)

Biological context: Proliferating stem cells with stochastic fate decisions.

```python
# Train RUOT model (Velocity + Growth + Score)

cb.tl.fit(
   adata=n_pc_adata,
    
   config="unbalanced_ot",   
    				#corresponding to mode="neural_ode"
					# Match biology: velocity (differentiation) + growth (proliferation)
    
    
   device="cuda",
)

# Inspect outputs

print(f"Latent velocity: {n_pc_adata.obsm['velocity_latent'].shape}")

print(f"Growth rates: {n_pc_adata.obsm['growth_rate'].shape}")
```



##### Example 2: Large-Scale Steady-State Tissue (Velocity Only)

Biological context: Adult liver tissue (stable cell numbers, minimal noise).

```python
# Train Dynamical OT model (Velocity only)

cb.tl.fit(

   adata=n_pc_adata,

   config="CRUFM",  #corresponding to mode="flow_matching"
    				# Match biology: velocity (differentiation) + growth (proliferation) + score (stochasticity)

   device="cuda",

)

print(f"Latent velocity: {n_pc_adata.obsm['velocity_latent'].shape}")

print(f"Growth rates: {n_pc_adata.obsm['growth_rate'].shape}")

print(f"Stochastic score: {n_pc_adata.obsm['score_latent'].shape}")
```



## 4. Model Selection Guide

### 4.1 Component Selection (Biology-Driven)

Components map to biological phenomena—**velocity is mandatory**; add others based on your system.

| Component                | Biological Meaning                                           | When to Include                                  |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| **Velocity** (Mandatory) | Instantaneous direction of cell state transitions (e.g., differentiation/activation) | Always (core of trajectory inference)            |
| **Growth**               | Proliferation/apoptosis rates (changes in cell population size over time) | - Developmental expansion  -Tumor proliferation  |
| **Score**                | Stochasticity in cell state transitions (biological noise)   | - Stem cell fate plasticity                      |
| **Interaction**          | Cell-cell communication effects                              | Tumor microenvironment (immune-cancer crosstalk) |



### 4.2 Training Mode Selection (Computation-Driven)

Choose between `flow_matching` (speed) and `neural_ode` (flexibility):

| Training Mode   | Speed                          | Flexibility                        | Best For                                                     | Limitations                                   |
| --------------- | ------------------------------ | ---------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| `flow_matching` | ⚡ Fast (simulation-free)       | ❌ Low (simple loss functions only) | - Large datasets - Simple dynamics (velocity + growth + score) | Struggles with complex dynamics (interaction) |
| `neural_ode`    | 🐢 Slow (numerical ODE solving) | ✅ High (all components/losses)     | - Small/medium datasets - Complex dynamics (velocity + growth + score + interaction) | Computationally expensive for large data      |



### 4.3 Model Selection Cheat Sheet

| Biological Scenario                                     | Components                            | Training Mode                 | CytoBridge Config |
| ------------------------------------------------------- | ------------------------------------- | ----------------------------- | ----------------- |
| Only considering Velocity                               | Velocity only                         | `neural_ode`                  | `dynamical_ot`    |
| Proliferation (no stochasticity)                        | Velocity + Growth                     | `neural_ode`                  | `unbalanced_ot`   |
| Proliferation + Stochasticity                           | Velocity + Growth + Score             | `neural_ode` +`flow_matching` | `ruot`            |
| Proliferation + Stochasticity                           | Velocity + Growth + Score             | `flow_matching`               | `crufm`           |
| proliferation + Stochasticity + cell-cell communication | Velocity + Growth + Score+Interaction | `neural_ode`                  | `cytobridge`      |



## 5. Optional Loss Functions 

Loss functions in CytoBridge quantify the mismatch between model predictions and biological data. The choice of loss depends on your **model components** (e.g., growth requires mass-aware losses) and **biological goals** (e.g., stochasticity requires score matching). Below is a breakdown of all supported losses.

### 5.1 Overview

Each loss addresses a specific biological or computational challenge. Here’s a high-level mapping to model components:

| Loss Type           | Target Challenge                                             | Essential model components     |
| ------------------- | ------------------------------------------------------------ | ------------------------------ |
| OT Loss             | Align cell state distributions across time                   | Velocity (all OT-based models) |
| Mass Loss           | Enforce consistent cell population sizes (proliferation/apoptosis) | Velocity + Growth              |
| Score Matching Loss | Model stochasticity in cell state transitions                | Velocity + Score               |
| Density Loss        | Align local density of cell states                           | Velocity  (neural ODE mode)    |
| PINN Loss           | Enforce physical constraints (e.g., continuity of cell density) | Velocity (neural ODE mode)     |



### 5.2 OT Loss (Optimal Transport Loss)

#### Core Function

Measures the "cost" of transporting cell states from one time point to another—critical for aligning distributions in velocity-based models. Supports both **balanced** (equal cell counts) and **unbalanced** (proliferation/apoptosis) scenarios.

#### Method Options & Use Cases

The `method` parameter in `calc_ot_loss` dictates how transport costs are computed. Choose based on data size and whether you need gradient flow (for training).

| Method                             | Balanced/Unbalanced | Gradient Support        | Speed |
| ---------------------------------- | ------------------- | ----------------------- | ----- |
| `emd_detach`                       | Balanced            | ❌ No (weights detached) | Slow  |
| `sinkhorn`                         | Balanced            | ✅ Yes                   | Fast  |
| `sinkhorn_detach`                  | Balanced            | ❌ No                    | Fast  |
| `sinkhorn_knopp_unbalanced`        | Unbalanced          | ✅ Yes                   | Fast  |
| `sinkhorn_knopp_unbalanced_detach` | Unbalanced          | ❌ No                    | Fast  |

#### Key Parameters

* `reg` (Sinkhorn methods): Regularization strength (default: 0.1). Increase for noisy data; decrease for sharper distributions.

* `reg_m` (Unbalanced Sinkhorn): Controls mass imbalance tolerance (default: 0.01). Larger values allow more extreme proliferation/apoptosis.

  

### 5.3 Mass Loss

#### Core Function

Ensures the model preserves **local cell population sizes**—critical for models with the `Growth` component (proliferation/apoptosis). It matches the mass (cell count) of predicted cells to real cells in local neighborhoods.

#### Use Cases

* Models with `Growth` (Unbalanced OT, RUOT, cytobridge).
* Datasets where cell proliferation/apoptosis is heterogeneous (e.g., tumor cores vs. edges).

#### Key Parameters

* `relative_mass`: The expected proportion of cells in the target time point (e.g., 0.5 if cell count doubles between time points).

* `global_mass`: If `True`, adds a global mass constraint (ensures total cell count matches); use for datasets with global proliferation.



### 5.4 Score Matching Loss

#### Core Function

Models **stochasticity** in cell state transitions—required for models with the `Score` component (RUOT, cytobridge). It trains the model to predict the "score" (gradient of the log density) of noisy cell states, capturing biological noise.

#### Use Cases

* Models with `Score` (RUOT, cytobridge).
* Datasets with stochastic cell fate decisions (e.g., stem cell differentiation, iPSC reprogramming).

#### Key Parameters

* `sigma`: Noise level added to cell states (default: 0.1). Match to the noise in your data (e.g., higher for heterogeneous datasets).

* `lambda_penalty`: Penalty for positive log-probability (default: 1.0). Prevents unphysical positive scores.

* `model`: Trained `DynamicalModel` object (must include a `score_net`).



### 5.5 Density Loss

#### Core Function

Aligns the **local density** of predicted cell states with real cells—useful for scenarios where cell state density carries biological meaning (e.g., dense cell clusters representing cell types).

#### Use Cases

* Models where cell clustering is important (e.g., identifying stable cell types in differentiation trajectories).
* Spatial transcriptomics data (where local cell density correlates with tissue structure).

#### Key Parameters

* `hinge_value`: Minimum distance threshold for density matching (default: 0.01). Values below this are ignored (reduces noise).

* `top_k`: Number of nearest neighbors to consider for local density (default: 5).

* `groups`: Optional cell groups (e.g., cell types) to compute density loss per group.



### 5.6 PINN Loss (Physics-Informed Neural Network Loss)

#### Core Function

Enforces **physical constraints** on cell dynamics—specifically the **continuity equation** (conservation of cell density over time). Ideal for models trained with `neural_ode` mode that require mechanistic rigor.

#### Use Cases

* Models with `Velocity`, `Growth`, or `Interaction` (cytobridge).
* Mechanistic studies where dynamics must follow physical laws (e.g., tissue growth, cell migration).

#### Key Parameters

* `sigma`: Noise level for density estimation (default: 0.1).

* `use_mass`: If `True`, includes the `Growth` component in the continuity equation.

* `use_interaction`: If `True`, includes the `Interaction` component (cell-cell communication) in the equation.




### Loss Selection Cheat Sheet

| Model Config                        | Required Losses                                | Optional Losses            | Training Mode                  |
| ----------------------------------- | ---------------------------------------------- | -------------------------- | ------------------------------ |
| `dynamical_ot` (Velocity only)      | `OT Loss`                                      | `Density Loss`             | `neural_ode`                   |
| `unbalanced_ot` (Velocity + Growth) | `OT Loss`+ `Mass Loss`                         | `Density Loss`             | `neural_ode`                   |
| `ruot` (Velocity + Growth + Score)  | `OT Loss`+ `Mass Loss` + `Score Matching Loss` | `PINN Loss`+`Density Loss` | `neural_ode` + `flow_matching` |
| `crufm` (Velocity + Growth + Score) | `Score Matching Loss`                          |                            | `flow_matching`                |
| `cytobridge` (All components)       | `OT Loss`+ `Mass Loss` + `Score Matching Loss` | `Density Loss`+`PINN Loss` | `neural_ode`                   |
