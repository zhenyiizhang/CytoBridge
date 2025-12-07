## Installation

CytoBridge is currently under active development. You can install it using the methods outlined below:

### Recommended Approach: Install via pip  

Install CytoBridge quickly and easily using pip:

```bash
pip install CytoBridge
```

### Developer Approach: Clone the Repository  

This approach is suitable for developers who want to modify the source code or contribute to the project:  

1. Clone the repository to your local machine  

```bash
git clone https://github.com/zhenyiizhang/CytoBridge.git
```

2. Set up a Conda environment to manage dependencies  
   Follow the steps below to create an isolated environment for CytoBridge:  

   (1) Create and activate a new Conda environment  

   ```bash
   conda create -n CytoBridge python=3.10 ipykernel -y
   conda activate CytoBridge
   ```

   (2) Navigate to the root directory of the cloned repository  

   ```bash
   cd path_to_CytoBridge
   ```

   (3) Install all required dependencies  

   ```bash
   pip install -r requirements.txt
   ```



## Update Log  

**CytoBridge 1.2** (2025-10-29)  

- `score` component released → stochastic dynamics ready  
- RUOT pipeline stable under default hyper-params  (config/ruot)
- New plots: velocity/score/V+S streams, 2-D score (density) and ODE/SDE trajectories.
- evuluation and `test.ipynb` refreshed with downstream examples

**CytoBridge 1.3** (2025-11-25)  

- `interaction` component released →  cell-cell communication. ready  
- Conditional Flow Matching and  CytoBrigde (interaction) pipeline stable under default hyper-params  (config/ruot)
- New plots: interaction_stream, interaction_potential, landscape , process_sde_classification and analyze_terminal_states
- `test` refreshed with downstream examples

**CytoBridge 1.4** (2025-12-01)

- **Debug and Optimization**: Improved preprocessing to better adapt to real-world datasets.
- **Documentation Enhancements**: Refined the installation and guide to provide clearer instructions and better user guidance.
