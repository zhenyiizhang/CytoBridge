import torch
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from CytoBridge.tl.methods import neural_ode_step, ODEFunc
from CytoBridge.utils.utils import sample,trace_df_dz,compute_integral
from CytoBridge.tl.losses import calc_ot_loss, calc_mass_loss, calc_score_matching_loss,Density_loss,calc_pinn_loss
from CytoBridge.tl import methods
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.flow_matching import SchrodingerBridgeConditionalFlowMatcher, ConditionalRegularizedUnbalancedFlowMatcher, get_batch_size, compute_uot_plans, get_batch_uot_fm
from CytoBridge.pl.plot import plot_interaction_potential_epoch
from CytoBridge.tl.analysis import simulate_trajectory
import math
import os
import ot
from torchdiffeq import odeint
from torch.optim.lr_scheduler import StepLR  # Import StepLR scheduler
class TrainingPipeline:
    def __init__(self, model, config, batch_size, device, data):  # Added 'data' parameter for initialization
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None  # Initialize scheduler variable
        self.device = device
        self.model.to(device)
        # Determine if mass component is used based on model configuration
        self.use_mass = 'growth' in self.config['model']['components']
        # Determine if score component is used based on model configuration
        self.use_score = 'score' in self.config['model']['components']
        # Determine if interaction component is used based on model configuration
        self.use_interaction = 'interaction' in self.config['model']['components']

        # Initialize ODE function (unified gradient calculation entry)
        self.ode_func = ODEFunc(
            model=self.model,
            sigma=config['training']['defaults'].get('sigma', 0.05),
            use_mass=self.use_mass,
            score_use=self.use_score,
            interaction_use=self.use_interaction
        )

        # New: Initialize variables required for train_score_model
        self.logger = self._setup_logger()  # Simple logger implementation
        # Get experiment directory from configuration (default to './results' if not specified)
        self.exp_dir = self.config.get('ckpt_dir', './results')
        os.makedirs(self.exp_dir, exist_ok=True)
        # Construct DataFrame from input data to fit the format required by train_score_model
        self.df = self._prepare_df(data)
        # Get sorted list of unique time points (grouped by 'samples' column)
        self.groups = sorted(self.df.samples.unique())

    def _setup_logger(self):
        """Simple logger implementation to replace the original logger"""

        class SimpleLogger:
            @staticmethod
            def info(msg):
                print(f"[INFO] {msg}")

        return SimpleLogger()

    def _prepare_df(self, data):
        """Construct DataFrame from input data to fit the format required by train_score_model
        
        Args:
            data: List of tensors where each element represents samples at a specific time point (shape: n_samples×2)
        
        Returns:
            pd.DataFrame: Combined DataFrame with columns 'x1', 'x2', and 'samples' (time point)
        """
        all_samples = []
        for t_idx, x in enumerate(data):
            x_np = x.cpu().detach().numpy()  # Convert tensor to numpy array
            # Construct DataFrame for current time point: columns = [x1, x2, samples (time point)]
            df_t = pd.DataFrame({
                'x1': x_np[:, 0],
                'x2': x_np[:, 1],
                # Assign current time point to all samples (dtype: float64 for consistency)
                'samples': np.full(x_np.shape[0], t_idx, dtype=np.float64)
            })
            all_samples.append(df_t)
        # Concatenate DataFrames from all time points and reset index
        return pd.concat(all_samples, ignore_index=True)

    # --------------------------
    # Main Modifications: Optimizer and Scheduler Setup
    # --------------------------
    def _setup_stage(self, stage_params):
        lr = stage_params['lr']
        print(f"\n====  {stage_params['name']}  ====")

        # Get flags for score network training from stage parameters
        train_strategy = str(stage_params.get('train_strategy', '')).lower()



        if not train_strategy or train_strategy == 'none':
            use_v = train_g = use_s = use_i = True          # 缺省策略：全训练
        else:
            use_v, train_g, use_s, use_i = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy, 'i' in train_strategy

        if stage_params.get('mode') ==  "neural_ode":
            train_s = False
            if use_v and use_s and use_i:
                train_s = True
            self.model.use_growth_in_ode_inter = stage_params.get('use_growth_in_ode_inter', True)
            self.ode_func.use_mass = self.model.use_growth_in_ode_inter
            self.ode_func.score_use = train_strategy is not None and 's' in train_strategy
            self.ode_func.interaction_use = train_strategy is not None and 'i' in train_strategy

        elif stage_params.get('mode') ==  "flow_matching":
            train_s = True
        else:
            raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        # Collect trainable parameters based on component flags
        params = []

        for name, module in self.model.named_children():
            # print(f"Name: {name}")
            # print(f"Module: {module}")
            # print(f"Module type: {type(module)}")
            # print("Parameters:")
            if (name == 'velocity_net' and use_v) or (name == 'growth_net' and train_g) or  (name == 'score_net' and train_s) or (name == 'interaction_net'  and use_i):
                for p in module.parameters():
                    p.requires_grad = True
                    params.append(p)
                    # print(f"  Parameter shape: {p.shape}")
                    # print(f"  Parameter requires_grad: {p.requires_grad}")
                print("-" * 50)
            else:

                for p in module.parameters():
                    p.requires_grad = False
                    # print(f"  Parameter shape: {p.shape}")
                    # print(f"  Parameter requires_grad: {p.requires_grad}")
        # Initialize Adam optimizer with only trainable parameters
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, params), lr=lr
        )

        # Reset scheduler before setting up new one
        self.scheduler = None
        if 'scheduler_type' in stage_params:
            if stage_params['scheduler_type'] == 'cosine':
                # Use Cosine Annealing scheduler if specified
                cosine_epochs = stage_params.get('cosine_epochs', 1000)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=cosine_epochs, 
                    eta_min=1e-5  # Minimum learning rate
                )
            elif stage_params['scheduler_type'] == 'steplr':
                # Use StepLR scheduler if specified
                self.scheduler = StepLR(
                    optimizer=self.optimizer,
                    step_size=stage_params['scheduler_step_size'],
                    gamma=stage_params['scheduler_gamma']  # Learning rate decay factor
                )
                print(f"  Enabled learning rate scheduler: step_size={stage_params['scheduler_step_size']}, gamma={stage_params['scheduler_gamma']}")
        else:
            print("  No scheduler parameters configured - keeping learning rate constant")

        # Print gradient status (trainable/non-trainable) for each module
        for n, m in self.model.named_children():
            flag = any(p.requires_grad for p in m.parameters())
            print(f"  {n:<15}  grad={flag}")
        # Print shapes of parameters in optimizer
        print("  Optimizer parameters (shapes):", [p.shape for g in self.optimizer.param_groups for p in g['params']])

    def train(self, data, time_points):
        """Main training loop that executes multiple training stages based on configuration
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            DynamicalModel: Trained model
        """
        # Get training plan and base default parameters from configuration
        training_plan = self.config['training']['plan']
        base_defaults = self.config['training']['defaults']

        # Execute each stage in the training plan
        for stage_config in training_plan:
            # Merge base defaults with stage-specific config (stage config takes priority)
            stage_params = base_defaults.copy()
            stage_params.update(stage_config)
            stage_name = stage_params['name']


            print(f"\n--- Starting Stage: {stage_name} ---")
            print(
                f"  Mode: {stage_params['mode']}, Epochs: {stage_params['epochs']}, Use Score: {stage_params.get('score_use', False)}")
            train_strategy = stage_params.get('train_strategy', None)


            # Setup optimizer, scheduler, and trainable parameters for current stage
            self._setup_stage(stage_params)

            # Execute stage training based on mode
            if stage_params['mode'] == 'neural_ode':
                self.run_neural_ode_stage(stage_params, data, time_points)
            elif stage_params['mode'] == 'flow_matching':
                self.run_flow_matching_stage(stage_params, data, time_points)
            else:
                raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        return self.model

    def run_neural_ode_stage(self, stage_params, data, time_points):
        """Execute training stage using Neural ODE mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        epochs = stage_params['epochs']
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')


        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state = self.model.state_dict()
        train_strategy = stage_params.get('train_strategy', None)
        train_name=stage_params["name"]

        # Training loop over epochs
        for epoch in range(epochs):
            # Calculate loss for one epoch of Neural ODE training
            loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"  Stage '{stage_params['name']}', Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
            # if epoch % 10 == 0 and self.use_interaction:
            #     plot_interaction_potential_epoch(self.model,d=1,num_points=40,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #     if epoch < 15:
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")

            # if "i" in train_strategy:
            #     if epoch % 10 == 0:
            #         plot_interaction_potential_epoch(self.model,d=1,num_points=21,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")
            # Update best model if current loss is lower than previous best
            if loss < best_loss:
                best_loss = loss
                self.logger.info(f"Epoch {epoch:3d} has a lower loss| all_loss {best_loss:.4f}")
                best_state = self.model.state_dict()

        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state
            save_loss = best_loss
        else:  # 'last' strategy
            save_state = self.model.state_dict()
            # Recalculate loss for last epoch to ensure accuracy
            last_loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)
            save_loss = last_loss

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best.pth' if save_strategy == 'best_model' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) saved → {ckpt_dir}/{save_strategy}.pth")

    def train_neural_ode_epoch(self, stage_params, data, time_points, ode_func):
        """Calculate loss for one epoch of Neural ODE training
        
        Args:
            stage_params: Dictionary of parameters for current stage (loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
            ode_func: ODEFunc instance for computing ODE updates
        
        Returns:
            float: Average loss over all time intervals
        """
        # Get loss weights and configuration from stage parameters
        lambda_ot = stage_params['lambda_ot']
        lambda_mass = stage_params['lambda_mass']
        lambda_energy = stage_params['lambda_energy']
        
        OT_loss_type = stage_params['OT_loss']
        use_density_loss = stage_params.get('use_density_loss', False)
        use_pinn_loss = stage_params.get('use_pinn_loss', False)

        global_mass = stage_params.get('global_mass', False)
        if use_density_loss:
            if 'density_top_k' not in stage_params or 'lambda_density' not in stage_params or 'density_hinge_value' not in stage_params:
                raise ValueError(
                    "When use_density_loss=True, all 'density_top_k','lambda_density' and 'density_hinge_value' "
                    "must be provided in stage_params.(Default recommended ( 5 , 10 and  0.01))" 
                )            
            top_k = stage_params['density_top_k']
            hinge_value = stage_params['density_hinge_value']
            lambda_density = stage_params['lambda_density']
            density_fn = Density_loss(hinge_value)


        # Initialize with sampled data from the first time point
        x0 = sample(data[0], self.batch_size).to(self.device)
        # Initialize log-weights (uniform distribution)
        lnw0 = torch.log(torch.ones(self.batch_size, 1) / self.batch_size).to(self.device)
        # Total number of samples at the first time point
        mass_0 = data[0].shape[0]

        total_loss = 0.0
        # Iterate over all time intervals (from t_{i-1} to t_i)
        for idx in range(1, len(time_points)):
            # Reset gradients before each time interval update
            self.optimizer.zero_grad()

            # Get current time interval and target data
            t0, t1 = time_points[idx - 1], time_points[idx]
            data_t1 = sample(data[idx], self.batch_size).to(self.device)
            # Total number of samples at the target time point
            mass_1 = data[idx].shape[0]
            # Calculate relative mass ratio between target and initial time points
            relative_mass = mass_1 / mass_0

            # Perform one Neural ODE step to predict state at t1
            x1, lnw1, e1 = neural_ode_step(ode_func, x0, lnw0, t0, t1, self.device)

            # Calculate individual loss components
            loss_ot = calc_ot_loss(x1, data_t1, lnw1, OT_loss_type)
            # Calculate mass loss only if mass component is enabled
            loss_mass = calc_mass_loss(x1, data_t1, lnw1, relative_mass, global_mass) if self.use_mass else 0.0
            # Energy loss (average of energy term from ODE step)
            loss_energy = e1.mean()

            # Combine losses with respective weights
            loss = (lambda_ot * loss_ot) + (lambda_mass * loss_mass) + (lambda_energy * loss_energy)

            if use_density_loss:          
                density_loss = density_fn(x1, data_t1, top_k=top_k)
                density_loss = density_loss.to(loss.device)
                loss += lambda_density * density_loss
                # print('density loss')
                # print(density_loss)
            if use_pinn_loss: 
                if 'lambda_pinn'  not in stage_params:
                    raise ValueError(
                        "When use_pinn_loss=True, 'lambda_pinn' must be provided in stage_params.(Default recommended (100))" 
                    )            
                lambda_pinn = stage_params['lambda_pinn'] 

                loss_pinn = calc_pinn_loss(self, t1, data_t1,sigma=stage_params['sigma'], use_mass=self.use_mass,trace_df_dz=trace_df_dz,device=self.device)
                # print("loss_pinn",loss_pinn)
                # print("loss",loss)
                loss += lambda_pinn * loss_pinn
            # print(f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy}), Density Loss: {density_loss:.4f} (λ={lambda_density})" if use_density_loss else f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy})", end="")
            # if use_pinn_loss:
            #     print(f", PINN Loss: {loss_pinn:.4f} (λ={lambda_pinn})")
            # Backpropagate gradients and update optimizer
            loss.backward()
            self.optimizer.step()

            # Update initial state for next time interval (detach to avoid gradient accumulation)
            x0 = x1.clone().detach()
            lnw0 = lnw1.clone().detach()

            # Accumulate total loss over all time intervals
            total_loss += loss.item()

        # Return average loss per time interval
        return total_loss / (len(time_points) - 1)


    def run_flow_matching_stage(self, stage_params, data, time_points):
        """Execute training stage using Flow Matching mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, sigma, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)

        # Convert time points to tensor (device-compatible)
        time = torch.tensor(time_points, device=self.device, dtype=torch.float32)
        # Get sigma parameter for Flow Matching
        sigma = stage_params['sigma']
        # Get alpha regularization parameter (default to 1.0 if not specified)
        alpha_regm = stage_params.get('alpha_regm', 1.0)
        print("alpha_regm :", alpha_regm)
        self.sigma = sigma
        # Convert data to list of numpy arrays (required for compute_uot_plans)
        X = [data[i].float().cpu().detach().numpy() for i in range(len(time_points))]
        
        # Get flags for training different network components
        train_strategy = str(stage_params.get('train_strategy', 's')).lower()
        regress_v, regress_g, regress_score = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy
        
        if regress_g or regress_v :
            uot_plans, sampling_info = compute_uot_plans(X, time_points,use_mini_batch_uot=True, chunk_size=1000, alpha_regm= alpha_regm ,reg_strategy="max_over_time")
        else :
            uot_plans, sampling_info = compute_uot_plans(X, time_points,use_mini_batch_uot=True, chunk_size=2000,reg_strategy='per_time')

        # Initialize Conditional Regularized Unbalanced Flow Matcher
        FM = ConditionalRegularizedUnbalancedFlowMatcher(sigma=sigma)
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')
        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state_dict = None
        
        # Get batch size from stage parameters
        batch_size = stage_params['batch_size']



        # Training loop over epochs (with tqdm progress bar)
        for epoch in tqdm(range(stage_params['epochs']), desc='Flow matching'):
            # Calculate loss for one epoch of Flow Matching training
            loss, penalty = self.train_flow_matching_epoch(
                FM, X, time,
                self.optimizer,
                stage_params['flow_matching']['lambda_penalty'],
                batch_size,
                uot_plans,
                sampling_info,
                regress_v, regress_g, regress_score,
            )

            # Stop training if loss becomes NaN (numerical instability)
            if torch.isnan(loss):
                self.logger.info("Training stopped due to NaN loss")
                # Load best model state before NaN occurred
                self.model.load_state_dict(best_state_dict)
                break

            # Update best model if current loss is lower than previous best
            if loss < best_loss:
                best_loss = loss
                best_state_dict = self.model.state_dict().copy()

            # Combine loss and penalty for backpropagation
            total_loss = loss + penalty
            # print("score_loss",loss)
            # print("penalty",penalty)

            total_loss.backward()
            # Update optimizer
            self.optimizer.step()
            # Update scheduler if initialized
            if self.scheduler is not None:
                self.scheduler.step()

        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state_dict
            save_loss = best_loss
        else:  # 'last' strategy
            save_state = self.model.state_dict()
            save_loss = loss.item() + penalty.item()

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) "
              f"saved → {ckpt_dir}/{save_strategy}_model.pth")

    def train_flow_matching_epoch(self, FM, X, time,
                                  optimizer, lambda_pen, batch_size, uot_plans, sampling_info, regress_v, regress_g, regress_score):
        """Calculate loss for one epoch of Flow Matching training
        
        Args:
            FM: ConditionalRegularizedUnbalancedFlowMatcher instance
            X: List of numpy arrays where each element represents samples at a specific time point
            time: Tensor of time points (device-compatible)
            optimizer: Torch optimizer instance
            lambda_pen: Penalty weight for score network training
            batch_size: Batch size for sampling
            uot_plans: Precomputed UOT plans for sampling
            sampling_info: Additional sampling information from compute_uot_plans
            regress_v: Flag to train velocity network (v)
            regress_g: Flag to train growth network (g)
            regress_score: Flag to train score network
        
        Returns:
            tuple: (total_loss, penalty) where both are torch tensors
        """
        # Reset gradients before each batch
        optimizer.zero_grad()
        # Sample batch data for Flow Matching (time, positions, velocities, growth values, weights, noise)
        t, xt, ut, gt_samp, weights, eps = get_batch_uot_fm(FM, X, time, batch_size, uot_plans, sampling_info)
        # Reshape time tensor to (batch_size, 1) for concatenation with position data
        t = torch.unsqueeze(t, 1).to(self.device)

        # Compute lambda(t) (time-dependent weighting factor for score network)
        t_floor = torch.zeros_like(t)
        t_ceil = torch.zeros_like(t)
        # Determine time interval bounds (t_floor and t_ceil) for each sample in the batch
        for j in range(len(time) - 1):
            mask = (t >= time[j]) & (t < time[j + 1])
            t_floor[mask] = time[j]
            t_ceil[mask] = time[j + 1]
        # Calculate normalized time within interval and compute lambda(t)
        lambda_t = FM.compute_lambda((t - t_floor) / (t_ceil - t_floor))

        # Enable gradient computation for position data (required for score calculation via autograd)
        xt = xt.requires_grad_(True)
        # Get references to model components
        v_net = self.model.velocity_net
        g_net = self.model.growth_net
        score_net = self.model.score_net
        # Concatenate position and time data for network input (shape: batch_size × (2 + 1) = batch_size × 3)
        net_input = torch.cat([xt, t], dim=1)

        # Initialize loss and penalty
        loss = 0.0
        penalty = 0.0
        # Train score network if enabled
        if regress_score:
            # Predict score potential (value_st) from score network
            value_st = score_net(net_input)
            # Compute score via automatic differentiation (gradient of value_st w.r.t. xt)
            st = torch.autograd.grad(
                outputs=value_st,
                inputs=xt,
                grad_outputs=torch.ones_like(value_st),
                create_graph=True  # Required for second-order gradients (if needed)
            )[0]
            # Calculate weighted MSE loss for score network
            score_loss = torch.mean(weights * ((lambda_t[:, None] * st + eps) ** 2))
            # Handle NaN loss (set to 0 to avoid training instability)
            if torch.isnan(score_loss):
                score_loss = 0.0
            loss += score_loss
            # Add penalty term to regularize score potential (prevents exploding values)
            penalty += lambda_pen * torch.max(torch.relu(value_st))
        
        # Train velocity network (v) if enabled
        if regress_v:
            # Predict velocity from velocity network
            v_predict = v_net(net_input)
            # Add weighted MSE loss between predicted and target velocities
            loss += torch.mean(weights * (v_predict - ut) ** 2)
        
        # Train growth network (g) if enabled
        if regress_g:
            # Predict growth values from growth network
            g_predict = g_net(net_input)
            # Add weighted MSE loss between predicted and target growth values (scaled by 1000 for better convergence)
            loss += 1000 * torch.mean(weights * (g_predict - gt_samp) ** 2)

        return torch.as_tensor(loss, device=self.device), torch.as_tensor(penalty, device=self.device)
    def evaluate(self,adata, data, time_points):
        """Evaluate trained model using Wasserstein-1 distance and Total Mass Variation (TMV)
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            list: List of Wasserstein-1 distances for each time point (excluding initial time)
        """
        print(f"\n--- Starting Evaluation ---")
        device = self.device
        # Get initial time point data (t=0)
        x0 = data[0].to(device)
        # Freeze model parameters during evaluation (disable gradient computation)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get sigma parameter (use stored value or default to 0.05 if not available)
        sigma = getattr(self, 'sigma', None) or 0.05

        # Simulate trajectory using the trained model
        point, weight = simulate_trajectory(
            adata,
            self.model,
            x0,
            sigma,           
            time_points,
            dt=0.01,  # Time step for ODE simulation
            device=x0.device
        )

        # Calculate Wasserstein-1 distance for each time point (excluding initial time)
        wasserstein_scores = []
        for idx in range(1, len(time_points)):
            t0, t1 = time_points[0], time_points[idx]
            # Get target data at current time point (convert to numpy for OT computation)
            data_t1 = data[idx].detach().cpu().numpy()
            # Get predicted positions and weights from simulated trajectory
            x1 = point[idx]
            m1 = weight[idx]

            # Calculate Total Mass Variation (TMV) between predicted and true mass
            tmv = np.abs(m1.sum() - data[idx].shape[0] / data[0].shape[0])
            # Normalize predicted weights to sum to 1 (required for OT)
            m1 = m1 / m1.sum()

            # Create uniform weights for target data (sum to 1)
            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
            # Compute Euclidean distance matrix between target and predicted points
            cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

            # Calculate Wasserstein-1 distance using Earth Mover's Distance (EMD)
            w1 = ot.emd2(
                m2,
                m1.reshape(-1),  # Reshape to 1D array (required by ot.emd2)
                cost_matrix,
                numItermax=1e7  # Increase max iterations for convergence
            )

            # Store results and print progress
            wasserstein_scores.append(w1)
            print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f}")
            print(f"  Time Point {t1}: TMV = {tmv:.4f}")
        
        return wasserstein_scores

    
    # def generate_state_trajectory(self, data, time_points):
    #     """Generate reference trajectory without score guidance (using only velocity and growth components)
        
    #     Args:
    #         data: List of tensors where each element represents samples at a specific time point
    #         time_points: List of time values corresponding to each element in 'data'
        
    #     Returns:
    #         list: List of tensors representing predicted positions at each time point (detached from graph)
    #     """
    #     # Get initial time point data (t=0)
    #     x0 = data[0].to(self.device)
    #     n_samples = x0.shape[0]

    #     # Initialize ODE function (force disable score component)
    #     ode_func = ODEFunc(
    #         model=self.model,
    #         sigma=self.config['training']['defaults'].get('sigma', 0.1),
    #         use_mass=self.use_mass,
    #         score_use=False,
    #         score_flow_matching_use=False
    #     ).to(self.device)

    #     # ODE initial state: (positions, log-weights, mass)
    #     init_lnw = torch.log(torch.ones(n_samples, 1) / n_samples).to(self.device)
    #     init_m = torch.zeros_like(init_lnw).to(self.device)
    #     initial_state = (x0, init_lnw, init_m)

    #     # Solve ODE to get full trajectory
    #     t_eval = torch.tensor(time_points, device=self.device, dtype=torch.float32)
    #     traj_x, _, _ = odeint(
    #         func=ode_func,
    #         y0=initial_state,
    #         t=t_eval,
    #         method='euler'  # Euler method for ODE solving (fast but less accurate)
    #     )

    #     # Split trajectory by time point and detach from computation graph (avoid memory leaks)
    #     return [traj_x[i].detach() for i in range(len(time_points))]


    # def generate_state_trajectory1(self, data, time_points, reg=None, reg_m=None, method='sinkhorn', numItermax=1000,
    #                                stopThr=1e-6, **kwargs):
    #     """Generate trajectory with Unbalanced Sinkhorn matching (fixed: ensure valid trajectory output + add error handling)
        
    #     Args:
    #         data: List of tensors where each element represents samples at a specific time point
    #         time_points: List of time values corresponding to each element in 'data'
    #         reg: Regularization parameter for Sinkhorn (auto-calculated if None)
    #         reg_m: Mass regularization parameter for Unbalanced Sinkhorn (auto-calculated if None)
    #         method: Matching method (default: 'sinkhorn')
    #         numItermax: Maximum number of iterations for Sinkhorn
    #         stopThr: Convergence threshold for Sinkhorn
    #         **kwargs: Additional keyword arguments
        
    #     Returns:
    #         list: List of tensors representing matched trajectory (detached from graph)
    #     """
    #     try:
    #         # 1. Get number of samples at each time point
    #         max_iter = numItermax
    #         tol = stopThr
    #         data_sizes = [d.shape[0] for d in data]
    #         raw_masses = data_sizes

    #         # 2. Determine sample sizes for each time point (balance between speed and accuracy)
    #         min_size = min(data_sizes)
    #         max_size = max(data_sizes)
    #         print("min_size", min_size, "max_size", max_size)
            
    #         if min_size >= 1024:
    #             # Scale sample sizes proportionally if minimum size is ≥1024
    #             sample_sizes = [max(1, int(round(1024 * s / min_size))) for s in data_sizes]
    #         elif max_size >= 1024:
    #             # Cap sample sizes at max_size if maximum size is ≥1024 (avoid oversampling)
    #             sample_sizes = []
    #             for s in data_sizes:
    #                 target = max(1, int(round(1024 * s / min_size)))
    #                 target = min(target, s)
    #                 sample_sizes.append(target)
    #         else:
    #             # Use original sample sizes if all are <1024
    #             sample_sizes = data_sizes

    #         # 3. Sample data for each time point (ensure consistent batch size)
    #         sampled_data = []
    #         for t_idx in range(len(time_points)):
    #             size = sample_sizes[t_idx]
    #             data_t = data[t_idx].to(self.device)
    #             # Oversample if current time point has fewer samples than target size
    #             if data_t.shape[0] < size:
    #                 indices = torch.randint(0, data_t.shape[0], (size,), device=self.device)
    #             else:
    #                 # Undersample if current time point has more samples than target size
    #                 indices = torch.randperm(data_t.shape[0], device=self.device)[:size]
    #             sampled = data_t[indices]
    #             sampled_data.append(sampled)

    #         # 4. Unbalanced Sinkhorn matching to connect time points
    #         matched_trajectories = [sampled_data[0]]  # Initialize trajectory with first time point

    #         # Iterate over time points to match consecutive time steps
    #         for t_idx in range(1, len(time_points)):
    #             t_prev = time_points[t_idx - 1]
    #             t_curr = time_points[t_idx]
    #             prev_points = matched_trajectories[-1]  # Points from previous time point
    #             curr_points = sampled_data[t_idx]  # Points from current time point
    #             n, m = prev_points.shape[0], curr_points.shape[0]

    #             # Convert tensors to numpy arrays (required for OT library)
    #             prev_np = prev_points.cpu().detach().numpy()
    #             curr_np = curr_points.cpu().detach().numpy()

    #             # Get true mass values for previous and current time points
    #             prev_mass = raw_masses[t_idx - 1]
    #             curr_mass = raw_masses[t_idx]

    #             # Auto-calculate regularization parameters if not provided
    #             if reg is None or reg_m is None:
    #                 auto_reg, auto_reg_m = self.calculate_auto_regularization(prev_np, curr_np, prev_mass, curr_mass)
    #                 reg = auto_reg if reg is None else reg
    #                 reg_m = auto_reg_m if reg_m is None else reg_m
    #                 print(f"Auto-calculated regularization: reg={reg:.4f}, reg_m={reg_m:.4f}")
    #             else:
    #                 print(f"User-specified regularization: reg={reg:.4f}, reg_m={reg_m:.4f}")

    #             # Predict source weights (a) using growth network (fixed: ensure correct weight calculation)
    #             # a. Initialize log-weights for previous time point (uniform distribution)
    #             lnw_prev_init = torch.log(torch.ones(n, 1, device=self.device) / n)
    #             # b. Define time interval for ODE solving (from previous to current time point)
    #             t_interval = torch.tensor([t_prev, t_curr], device=self.device, dtype=torch.float32)
    #             # c. Initialize ODE state (positions, log-weights, mass)
    #             initial_state = (prev_points, lnw_prev_init, torch.zeros_like(lnw_prev_init, device=self.device))
    #             # d. Solve ODE to get predicted log-weights at current time point
    #             traj_x, traj_lnw, _ = odeint(
    #                 func=self.ode_func,
    #                 y0=initial_state,
    #                 t=t_interval,
    #                 method='euler'
    #             )
    #             # e. Convert log-weights to probabilities (normalize to sum to 1)
    #             lnw_prev_pred = traj_lnw[-1]
    #             mu_prev = torch.exp(lnw_prev_pred)
    #             mu_prev = mu_prev / mu_prev.sum()
    #             a = mu_prev.cpu().detach().numpy().squeeze()  # Source weights (1D array)

    #             # Target weights (b): uniform distribution over current time point samples
    #             nu_curr = torch.ones(m, 1, device=self.device) / m
    #             b = nu_curr.cpu().detach().numpy().squeeze()  # Target weights (1D array)

    #             # Compute Euclidean distance matrix between previous and current points
    #             M = ot.dist(prev_np, curr_np)
    #             # Solve Unbalanced Sinkhorn to get transport matrix
    #             transport_matrix = ot.unbalanced.sinkhorn_unbalanced(
    #                 a, b, M, reg, reg_m,
    #                 numItermax=max_iter, stopThr=tol
    #             )

    #             # Match current time point points to previous time point (max weight in transport matrix)
    #             sinkhorn_result = torch.tensor(transport_matrix, device=self.device)
    #             matched_indices = torch.argmax(sinkhorn_result, dim=1)  # For each previous point, find best current point
    #             matched_points = curr_points[matched_indices]
    #             matched_trajectories.append(matched_points)

    #         # Ensure trajectory is not empty (raise error if no points were generated)
    #         if not matched_trajectories:
    #             raise ValueError("Trajectory generation failed: no points were generated")

    #         # Detach all points from computation graph and return trajectory
    #         trajectory = [points.detach() for points in matched_trajectories]
    #         return trajectory

    #     except Exception as e:
    #         # Print error message and return original data (detached) as fallback
    #         print(f"Trajectory generation encountered an error: {str(e)}")
    #         return [data[t_idx].detach() for t_idx in range(len(time_points))]


    # def visualize_trajectory(self, trajectory, trajectory_times):
    #     """Visualize trajectory with scatter plots (time-colored) and connecting lines for each trajectory chain
        
    #     Args:
    #         trajectory: List of tensors where each element represents predicted positions at a specific time point
    #         trajectory_times: List of time values corresponding to each element in 'trajectory'
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     # 1. Prepare data for scatter plot (combine all time points)
    #     all_data = np.concatenate([x.cpu().detach().numpy() for x in trajectory], axis=0)
    #     # Create time labels for color coding (each sample gets its corresponding time point)
    #     time_labels = np.concatenate([np.full(x.shape[0], t.item()) for x, t in zip(trajectory, trajectory_times)])

    #     # 2. Create plot and scatter plot (time-colored points)
    #     plt.figure(figsize=(8, 6))
    #     # Scatter plot: color by time point, semi-transparent, higher z-order (on top of lines)
    #     scatter = plt.scatter(
    #         all_data[:, 0],
    #         all_data[:, 1],
    #         c=time_labels,
    #         cmap='viridis',
    #         alpha=0.6,
    #         zorder=2
    #     )
    #     # Add color bar to indicate time point mapping
    #     plt.colorbar(scatter, label='Time Point')

    #     # 3. Add connecting lines for each trajectory chain (same sample across time points)
    #     # Reshape trajectory to (num_time_points, num_trajectories, 2) for easy indexing
    #     traj_matrix = np.concatenate([
    #         pts.cpu().detach().numpy()[:, :2][None, ...]  # Shape: (1, num_trajectories, 2)
    #         for pts in trajectory
    #     ], axis=0)  # Final shape: (num_time_points, num_trajectories, 2)
    #     T, n_traj = traj_matrix.shape[:2]

    #     # Plot line for each trajectory chain (low alpha + low z-order to not obscure scatter points)
    #     for traj_id in range(n_traj):
    #         plt.plot(
    #             traj_matrix[:, traj_id, 0],  # X-coordinates across time
    #             traj_matrix[:, traj_id, 1],  # Y-coordinates across time
    #             color='black',
    #             linewidth=0.8,
    #             alpha=0.4,
    #             zorder=1
    #         )

    #     # Add plot labels and title
    #     plt.xlabel('Latent Dimension 1')
    #     plt.ylabel('Latent Dimension 2')
    #     plt.title('Trajectory Visualization (lines connect same chain)')
    #     # Save plot (high DPI for clarity, tight layout to avoid label cutoff)
    #     plt.savefig("/home/sjt/workspace2/CytoBridge_test_main/figures/tra_test.png", dpi=300, bbox_inches='tight')


    # def _plot_snapshot(self, epoch, stage_params, data, time_points, exp_fig_dir):
    #     """Plot SDE trajectory and score field at specific time points for current epoch
        
    #     Args:
    #         epoch: Current training epoch (for filename labeling)
    #         stage_params: Stage-specific parameters (not used directly but kept for consistency)
    #         data: Input data (not used directly but kept for consistency)
    #         time_points: List of time points (not used directly but kept for consistency)
    #         exp_fig_dir: Directory to save plot files
    #     """
    #     # Create directory for figures if it doesn't exist
    #     os.makedirs(exp_fig_dir, exist_ok=True)

    #     # 3. Plot score field for time points t=0,1,2,3,4
    #     for t in [0, 1, 2, 3, 4]:
    #         # Define save path with epoch and time point labels
    #         save_score = os.path.join(exp_fig_dir, f"score_epoch{epoch}_t{t}.png")
    #         # Generate and save score field plot
    #         plot_score_and_gradient(
    #             dynamical_model=self.model,
    #             device=self.device,
    #             t_value=float(t),  # Time point to visualize
    #             x_range=(0, 2.5),  # X-axis range for grid
    #             y_range=(0, 2.5),  # Y-axis range for grid
    #             save_path=save_score,
    #             cmap='rainbow'  # Color map for score visualization
    #         )