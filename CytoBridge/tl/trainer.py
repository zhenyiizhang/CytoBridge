import torch
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from CytoBridge.tl.methods import neural_ode_step, ODEFunc
from CytoBridge.utils.utils import sample
from CytoBridge.tl.losses import calc_ot_loss, calc_mass_loss, calc_score_matching_loss
from CytoBridge.tl import methods
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.flow_matching import SchrodingerBridgeConditionalFlowMatcher, ConditionalRegularizedUnbalancedFlowMatcher, get_batch_size, compute_uot_plans, get_batch_uot_fm
from CytoBridge.pl.plot import plot_score_and_gradient
from CytoBridge.tl.analysis import simulate_trajectory
import math
import os
import ot
from torchdiffeq import odeint
from torch.optim.lr_scheduler import StepLR  # Import StepLR for learning rate decay


class TrainingPipeline:
    """
    A pipeline class to manage end-to-end training of dynamical models (e.g., DynamicalModel in CytoBridge).
    It supports multiple training stages (Neural ODE, Flow Matching), parameter configuration, model saving,
    and evaluation (Wasserstein distance calculation, trajectory simulation).
    
    Attributes:
        model: The dynamical model to train (contains velocity_net, growth_net, score_net).
        config: Training configuration dictionary (includes training plan, hyperparameters, save paths).
        batch_size: Batch size for training.
        device: Device (CPU/GPU) used for model training.
        optimizer: PyTorch optimizer (e.g., Adam) for parameter updates.
        scheduler: Learning rate scheduler (e.g., CosineAnnealingLR, StepLR) (optional).
        ode_func: ODEFunc instance to unify gradient computation for Neural ODE.
        use_mass: Boolean indicating if mass conservation (growth component) is enabled.
        use_score: Boolean indicating if score matching component is enabled.
        logger: Simple logger for training information output.
        exp_dir: Experiment directory to save checkpoints and results.
        df: DataFrame formatted from input data, used for score model training.
        groups: Sorted unique time points (from df['samples']) for data grouping.
    """
    def __init__(self, model, config, batch_size, device, data):  # 
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None  # Initialize scheduler as None (set later in stages)
        self.device = device
        self.model.to(device)  # Move model to the specified device

        # Determine if mass conservation (growth) and score components are used
        self.use_mass = 'growth' in self.config['model']['components']
        self.use_score = 'score' in self.config['model']['components']

        # Initialize ODE function (unified entry for gradient computation)
        self.ode_func = ODEFunc(
            model=self.model,
            sigma=config['training']['defaults'].get('sigma', 0.05),  # Noise scale (default: 0.05)
            use_mass=self.use_mass,
            score_use=self.use_score
        )

        # New: Initialize variables required for train_score_model
        self.logger = self._setup_logger()  # Create a simple logger (replaces external logger)
        self.exp_dir = self.config.get('ckpt_dir', './results')  # Get experiment dir from config (default: ./results)
        os.makedirs(self.exp_dir, exist_ok=True)  # Create dir if it doesn't exist
        self.df = self._prepare_df(data)  # Format input data into DataFrame
        self.groups = sorted(self.df.samples.unique())  # Get sorted unique time points

    def _setup_logger(self):
        """
        Create a simple logger class to replace external logger dependencies.
        Only supports INFO-level messages (printed to console with [INFO] prefix).
        """

        class SimpleLogger:
            @staticmethod
            def info(msg):
                print(f"[INFO] {msg}")

        return SimpleLogger()

    def _prepare_df(self, data):
        """
        Format input time-series data into a DataFrame for score model training.
        
        Args:
            data: A list of tensors, where each tensor represents samples at a single time point (shape: n_samples × 2).
        
        Returns:
            pd.DataFrame: Combined DataFrame with columns 'x1' (1st feature), 'x2' (2nd feature), 
                          and 'samples' (time point, stored as float64).
        """
        all_samples = []
        for t_idx, x in enumerate(data):
            x_np = x.cpu().detach().numpy()  # Convert tensor to NumPy array
            # Create DataFrame for the current time point
            df_t = pd.DataFrame({
                'x1': x_np[:, 0],  # 1st dimension of samples
                'x2': x_np[:, 1],  # 2nd dimension of samples
                'samples': np.full(x_np.shape[0], t_idx, dtype=np.float64)  # Time point (float64 for consistency)
            })
            all_samples.append(df_t)
        return pd.concat(all_samples, ignore_index=True)  # Combine all time points into one DataFrame

    # --------------------------
    # Main Modification: Optimizer and Scheduler Setup
    # --------------------------
    def _setup_stage(self, stage_params):
        """
        Configure optimizer, learning rate scheduler, and parameter gradient flags for the current training stage.
        
        Args:
            stage_params: Dictionary of parameters for the current stage (includes lr, scheduler type, score_use, etc.).
        """
        lr = stage_params['lr']  # Get learning rate for the stage
        print(f"\n====  {stage_params['name']}  ====")

        # Determine if score_net and its Flow Matching mode are enabled
        score_use = stage_params.get('score_use', False)
        score_fm_use = stage_params.get('score_train', False)

        # Collect trainable parameters based on component flags
        params = []
        for name, module in self.model.named_children():
            if name == 'score_net':
                # Enable gradients for score_net only if both score_use and score_fm_use are True
                if score_use and score_fm_use:
                    for p in module.parameters():
                        p.requires_grad = True
                        params.append(p)
                else:
                    # Disable gradients for score_net
                    for p in module.parameters():
                        p.requires_grad = False

            else:
                # Enable gradients for other modules (e.g., velocity_net, growth_net)
                for p in module.parameters():
                    p.requires_grad = True
                    params.append(p)
        
        # Initialize Adam optimizer (filter out parameters with no gradients)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, params), lr=lr
        )

        self.scheduler = None  # Reset scheduler for the new stage
        if 'scheduler_type' in stage_params:
            # Configure CosineAnnealingLR scheduler
            if stage_params['scheduler_type'] == 'cosine':
                cosine_epochs = stage_params.get('cosine_epochs', 3000)  # Default: 3000 epochs
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=cosine_epochs,  # Maximum number of iterations
                    eta_min=1e-5  # Minimum learning rate
                )

            # Configure StepLR scheduler
            elif stage_params['scheduler_type'] == 'steplr':
                self.scheduler = StepLR(
                    optimizer=self.optimizer,
                    step_size=stage_params['scheduler_step_size'],  # Epochs between lr decays
                    gamma=stage_params['scheduler_gamma']  # Decay factor
                )
                print(f"  Enabled learning rate scheduler: step_size={stage_params['scheduler_step_size']}, gamma={stage_params['scheduler_gamma']}")
        else:
            print("  No scheduler configured; learning rate remains constant")

        # Print gradient status for each model component
        for n, m in self.model.named_children():
            flag = any(p.requires_grad for p in m.parameters())
            print(f"  {n:<15}  grad={flag}")
        # Print shapes of trainable parameters (for verification)
        print("  Optimizer parameters: ", [p.shape for g in self.optimizer.param_groups for p in g['params']])

    def train(self, data, time_points):
        """
        Main training entry point: execute each stage in the training plan sequentially.
        
        Args:
            data: List of tensors, each representing samples at a time point (shape: n_samples × 2).
            time_points: List of time values corresponding to the data list.
        
        Returns:
            The trained dynamical model.
        """
        training_plan = self.config['training']['plan']  # Get list of training stages
        base_defaults = self.config['training']['defaults']  # Get base hyperparameters

        for stage_config in training_plan:
            # Merge base defaults with stage-specific parameters (stage params override defaults)
            stage_params = base_defaults.copy()
            stage_params.update(stage_config)
            stage_name = stage_params['name']

            print(f"\n--- Starting Stage: {stage_name} ---")
            print(
                f"  Mode: {stage_params['mode']}, Epochs: {stage_params['epochs']}, Use Score: {stage_params.get('score_use', False)}")

            # 1. Update ODE function parameters (ensure latest stage config is applied)
            self.ode_func.score_use = stage_params.get('score_use', False)
            self.ode_func.score_flow_matching_use = stage_params.get('score_train', False)

            # 2. Critical Fix: Configure optimizer/scheduler for ALL stages (ensures gradient flags are applied)
            self._setup_stage(stage_params)

            # 3. Execute training for the current stage mode
            if stage_params['mode'] == 'neural_ode':
                self.run_neural_ode_stage(stage_params, data, time_points)
            elif stage_params['mode'] == 'flow_matching':
                self.run_flow_matching_stage(stage_params, data, time_points)
            else:
                raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        return self.model

    def run_neural_ode_stage(self, stage_params, data, time_points):
        """
        Execute the Neural ODE training stage: manage epoch loop, best model tracking, and model saving.
        
        Args:
            stage_params: Stage-specific parameters (epochs, save_strategy, loss weights, etc.).
            data: List of tensors for each time point.
            time_points: List of time values corresponding to data.
        """
        epochs = stage_params['epochs']
        # Get save strategy (default: save the best model based on loss)
        save_strategy = stage_params.get('save_strategy', 'best')

        self._setup_stage(stage_params)  # Re-initialize stage (redundant but safe for consistency)

        best_loss = float('inf')  # Track the lowest loss for best model selection
        best_state = self.model.state_dict()  # Store weights of the best model
        
        for epoch in range(epochs):
            # Compute loss for one epoch of Neural ODE training
            loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"  Stage '{stage_params['name']}', Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

            # Update best model if current loss is lower
            if loss < best_loss:
                best_loss = loss
                self.logger.info(f"Epoch {epoch:3d} has a lower loss| all_loss {best_loss:.4f}")
                best_state = self.model.state_dict()  # Save new best weights

        # Select which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state
            save_loss = best_loss
        else:  # 'last' strategy: save the model from the final epoch
            save_state = self.model.state_dict()
            # Recompute loss for the final epoch (to ensure accuracy)
            last_loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)
            save_loss = last_loss

        # Load the selected state (best/last) and save to disk
        self.model.load_state_dict(save_state)
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])  # Checkpoint dir for the stage
        os.makedirs(ckpt_dir, exist_ok=True)
        # Determine checkpoint filename (fix: use 'best_model.pth' for 'best' strategy, 'last_model.pth' otherwise)
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) saved → {ckpt_dir}/{ckpt_filename}")

    def train_neural_ode_epoch(self, stage_params, data, time_points, ode_func):
        """
        Train one epoch of the Neural ODE model: compute loss (OT, mass, energy) and update parameters.
        
        Args:
            stage_params: Stage-specific parameters (loss weights, OT type, etc.).
            data: List of tensors for each time point.
            time_points: List of time values corresponding to data.
            ode_func: ODEFunc instance for Neural ODE step computation.
        
        Returns:
            float: Average loss over all time intervals (normalized by number of intervals).
        """
        # Get loss weights and configuration from stage parameters
        lambda_ot = stage_params['lambda_ot']  # Weight for OT loss
        lambda_mass = stage_params['lambda_mass']  # Weight for mass loss
        lambda_energy = stage_params['lambda_energy']  # Weight for energy loss
        global_mass = stage_params['global_mass']  # Flag for global mass conservation
        OT_loss_type = stage_params['OT_loss']  # Type of OT loss (e.g., 'emd', 'sinkhorn')

        # Initialize with sampled data from the first time point
        x0 = sample(data[0], self.batch_size).to(self.device)  # Initial samples (batch size)
        # Initial log-weights (uniform distribution: ln(1/batch_size))
        lnw0 = torch.log(torch.ones(self.batch_size, 1) / self.batch_size).to(self.device)
        mass_0 = data[0].shape[0]  # Total number of samples at t=0 (for relative mass calculation)

        total_loss = 0.0
        # Iterate over each time interval (t0 → t1)
        for idx in range(1, len(time_points)):
            self.optimizer.zero_grad()  # Reset gradients

            t0, t1 = time_points[idx - 1], time_points[idx]  # Current time interval
            data_t1 = sample(data[idx], self.batch_size).to(self.device)  # Samples at t1 (batch size)
            mass_1 = data[idx].shape[0]  # Total samples at t1
            relative_mass = mass_1 / mass_0  # Relative mass change between t0 and t1

            # Compute Neural ODE step: predict (x1, lnw1, energy) at t1 from t0
            x1, lnw1, e1 = neural_ode_step(ode_func, x0, lnw0, t0, t1, self.device)

            # Calculate individual losses
            loss_ot = calc_ot_loss(x1, data_t1, lnw1, OT_loss_type)  # OT loss (predicted vs real samples)
            # Mass loss (only if mass conservation is enabled)
            loss_mass = calc_mass_loss(x1, data_t1, lnw1, relative_mass, global_mass) if self.use_mass else 0.0
            loss_energy = e1.mean()  # Average energy loss from ODE step

            # Total loss (weighted sum of individual losses)
            loss = (lambda_ot * loss_ot) + (lambda_mass * loss_mass) + (lambda_energy * loss_energy)

            loss.backward()  # Backpropagate gradients
            self.optimizer.step()  # Update model parameters

            # Update initial state for next time interval (detach to avoid gradient accumulation)
            x0 = x1.clone().detach()
            lnw0 = lnw1.clone().detach()

            total_loss += loss.item()  # Accumulate loss over all intervals

        # Return average loss per time interval
        return total_loss / (len(time_points) - 1)


    def run_flow_matching_stage(self, stage_params, data, time_points):
        """
        Execute the Flow Matching training stage: compute UOT plans, initialize Flow Matcher,
        run epoch loop, and save the model.
        
        Args:
            stage_params: Stage-specific parameters (epochs, sigma, regularization, etc.).
            data: List of tensors for each time point.
            time_points: List of time values corresponding to data.
        """
        # Create checkpoint directory for the Flow Matching stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)

        # Remove pretrained model loading (ensure each component works independently)

        # # ---------- 1. Load pretrained weights (non-score components) ----------
        # # Get save strategy of the previous stage (default: 'best')
        # prev_stage   = stage_params.get('prev_stage', 'Pretrain')
        # prev_save_as = self.config.get('training', {})\
        #                         .get(prev_stage, {})\
        #                         .get('save_strategy', 'best')          
        # prev_ckpt_fn = 'best_model.pth' if prev_save_as == 'best' else 'last_model.pth'
        # best_prev_path = os.path.join(self.config.get('ckpt_dir', '.'), prev_stage, prev_ckpt_fn)

        # if os.path.exists(best_prev_path):
        #     self.model.load_state_dict(torch.load(best_prev_path, map_location=self.device))
        #     self.logger.info(f"Loaded pretrained model from {best_prev_path}")
        # else:
        #     self.logger.info(f"No pretrained checkpoint found at {best_prev_path}, train from scratch.")

        # Convert time points to tensor (for Flow Matching computations)
        time = torch.tensor(time_points, device=self.device, dtype=torch.float32)
        sigma = stage_params['sigma']  # Noise scale for Flow Matching
        alpha_regm = stage_params.get('alpha_regm', 1.0)  # Regularization parameter for UOT
        self.sigma = sigma  # Store sigma as instance attribute (for evaluation)
        # Convert data to list of NumPy arrays (required for compute_uot_plans)
        X = [data[i].float().cpu().detach().numpy() for i in range(len(time_points))]
        
        # Compute UOT (Unbalanced Optimal Transport) plans
        regress_g = stage_params.get('g_train', False)  # Flag for growth net training
        if regress_g:
            # Compute UOT plans with alpha regularization (for growth net)
            uot_plans, sampling_info = compute_uot_plans(
                X, time_points, 
                use_mini_batch_uot=True, 
                chunk_size=1000, 
                alpha_regm=alpha_regm, 
                reg_strategy='per_time'
            )
        else:
            # Compute UOT plans without alpha regularization
            uot_plans, sampling_info = compute_uot_plans(
                X, time_points, 
                use_mini_batch_uot=True, 
                chunk_size=2000, 
                reg_strategy='per_time'
            )



        # def rebuild_uot_plan(sampling_info: dict):
        #     """
        #     Reconstruct the full UOT plan matrix from chunked sub-plans (from sampling_info).
            
        #     Args:
        #         sampling_info: Dictionary containing chunked source/target indices and sub-plans.
            
        #     Returns:
        #         np.ndarray: Full UOT plan matrix (shape: n_source × n_target).
        #     """
        #     # Calculate total number of source and target samples
        #     n_source = sum(len(src_idx) for src_idx in sampling_info['source_groups'])
        #     n_target = sum(len(tgt_idx) for tgt_idx in sampling_info['target_groups'])

        #     G = np.zeros((n_source, n_target), dtype=np.float32)  # Initialize full plan matrix
        #     # Fill sub-plans into the full matrix using indices
        #     for src_idx, tgt_idx, sub_plan in zip(
        #         sampling_info['source_groups'],
        #         sampling_info['target_groups'],
        #         sampling_info['sub_plans']
        #     ):
        #         G[np.ix_(src_idx, tgt_idx)] = sub_plan
        #     return G

        # # Store reconstructed UOT plans (order matches uot_plans)
        # uot_plans_rebuilt = []

        # for si in sampling_info:  # si is either None (full matrix) or a dict (chunked)
        #     if si is None:  # UOT plan was computed as a full matrix
        #         uot_plans_rebuilt.append(None)
        #     else:  # UOT plan was chunked: reconstruct full matrix
        #         G = rebuild_uot_plan(si)  # Pass only the dict (si)
        #         uot_plans_rebuilt.append(G)

        # # Calculate row sums of reconstructed UOT plans (for verification)
        # row_sums = []
        # for mat in uot_plans_rebuilt:  # mat is np.ndarray (shape: n_i × n_j)
        #     row_sum = mat.sum(axis=1)  # Sum of each row (shape: n_i,)
        #     sum12 = row_sum.sum()  # Total sum of all rows
        #     mean12 = row_sum.mean()  # Average row sum

        #     row_sums.append(row_sum)


        # Initialize Conditional Regularized Unbalanced Flow Matcher
        FM = ConditionalRegularizedUnbalancedFlowMatcher(sigma=sigma)
        save_strategy = stage_params.get('save_strategy', 'best')  # Save strategy for the stage
        best_loss = float('inf')  # Track lowest loss for best model
        best_state_dict = None  # Store weights of the best model
        
        batch_size = stage_params['batch_size']  # Batch size for Flow Matching

        # Flags for training different model components
        regress_v = stage_params.get('v_train', False)  # Velocity net training flag
        regress_g = stage_params.get('g_train', False)  # Growth net training flag
        regress_score = stage_params.get('score_train', True)  # Score net training flag

        # Epoch loop for Flow Matching (with tqdm progress bar)
        for epoch in tqdm(range(stage_params['epochs']), desc='Flow matching'):
            # Compute loss and penalty for one Flow Matching epoch
            loss, penalty = self.train_flow_matching_epoch(
                FM, X, time,
                self.optimizer,
                stage_params['flow_matching']['lambda_penalty'],  # Penalty weight for score net
                batch_size,
                uot_plans,
                sampling_info,
                regress_v, regress_g, regress_score,
            )

            # Stop training if loss is NaN (numerical instability)
            if torch.isnan(loss):
                self.logger.info("Training stopped due to NaN loss")
                self.model.load_state_dict(best_state_dict)  # Revert to best model
                break

            # Update best model if current loss is lower
            if loss < best_loss:
                best_loss = loss
                best_state_dict = self.model.state_dict().copy()  # Save new best weights

            # # ---- Log progress ----
            # if epoch % 2999 == 0:
            #     current_lr = self.optimizer.param_groups[0]['lr']
            #     print(f"penalty: {penalty}")
            #     print(f"Iteration {epoch}: loss = {loss.item():.4f}, LR: {current_lr:.6f}")
            #     self._plot_snapshot(epoch, stage_params, data, time_points, "figures")

            # ---- Backward pass and parameter update ----
            total_loss = loss + penalty  # Total loss (loss + penalty)
            total_loss.backward()  # Backpropagate gradients
            self.optimizer.step()  # Update model parameters
            if self.scheduler is not None:
                self.scheduler.step()  # Update learning rate (if scheduler exists)

        # ---------- 5. Save model based on strategy ----------
        if save_strategy == 'best':
            save_state = best_state_dict
            save_loss = best_loss
        else:  # 'last' strategy: save final epoch model
            save_state = self.model.state_dict()
            save_loss = loss.item() + penalty.item()  # Total loss of final epoch

        # Load selected state (best/last) and save to disk
        self.model.load_state_dict(save_state)
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) "
              f"saved → {ckpt_dir}/{ckpt_filename}")

    def train_flow_matching_epoch(self, FM, X, time,
                             optimizer, lambda_pen, batch_size, uot_plans, sampling_info, regress_v, regress_g, regress_score):
        """
        Train one epoch of the Flow Matching model: sample batches, compute lambda(t),
        predict model outputs (score, velocity, growth), and calculate loss/penalty.
        
        Args:
            FM: ConditionalRegularizedUnbalancedFlowMatcher instance.
            X: List of NumPy arrays (samples at each time point).
            time: Tensor of time points (device: self.device).
            optimizer: PyTorch optimizer for parameter updates.
            lambda_pen: Weight for the score net penalty term.
            batch_size: Batch size for sampling.
            uot_plans: Precomputed UOT plans.
            sampling_info: Sampling information for UOT plans (chunked indices).
            regress_v: Boolean to enable velocity net training.
            regress_g: Boolean to enable growth net training.
            regress_score: Boolean to enable score net training.
        
        Returns:
            tuple: (loss, penalty) → Total component loss and score net penalty.
        """
        # 1. Reset gradients and sample a batch of data
        optimizer.zero_grad()
        t, xt, ut, gt_samp, weights, eps = get_batch_uot_fm(
            FM, X, time, batch_size, uot_plans, sampling_info
        )
        t = torch.unsqueeze(t, 1).to(self.device)  # Reshape time to (B, 1) for model input

        # 2. Compute lambda(t) (interpolation weight for Flow Matching)
        t_floor = torch.zeros_like(t)  # Lower bound of time interval for each sample
        t_ceil = torch.zeros_like(t)   # Upper bound of time interval for each sample
        for j in range(len(time) - 1):
            # Mask samples belonging to the j-th time interval [time[j], time[j+1])
            mask = (t >= time[j]) & (t < time[j + 1])
            t_floor[mask] = time[j]
            t_ceil[mask] = time[j + 1]
        # Compute lambda(t) using normalized time within the interval
        lambda_t = FM.compute_lambda((t - t_floor) / (t_ceil - t_floor))

        # 3. Score net prediction (requires autograd for gradient calculation)
        xt = xt.requires_grad_(True)  # Enable gradients for xt (to compute score via autograd)
        # Get model components
        v_net = self.model.velocity_net
        g_net = self.model.growth_net
        score_net = self.model.score_net
        # Concatenate xt (samples) and t (time) for model input (shape: B × (2 + 1) = B × 3)
        net_input = torch.cat([xt, t], dim=1)
        loss = 0.0
        penalty = 0.0

        # Calculate score net loss and penalty (if enabled)
        if regress_score:
            value_st = score_net(net_input)  # Predict score potential (value function)
            # Compute score via autograd (gradient of value_st w.r.t. xt)
            st = torch.autograd.grad(
                outputs=value_st,
                inputs=xt,
                grad_outputs=torch.ones_like(value_st),
                create_graph=True  # Enable higher-order gradients
            )[0]
            # Score loss (weighted MSE between predicted score and target)
            score_loss = torch.mean(weights * ((lambda_t[:, None] * st + eps) ** 2))
            if torch.isnan(score_loss):  # Avoid NaN loss
                score_loss = 0.0
            loss += score_loss
            # Penalty term (L1 penalty on positive score potential)
            penalty += lambda_pen * torch.max(torch.relu(value_st))
        
        # Calculate velocity net loss (if enabled)
        if regress_v:
            v_predict = v_net(net_input)  # Predict velocity
            # Weighted MSE loss between predicted and target velocity
            loss += torch.mean(weights * (v_predict - ut) ** 2)
        
        # Calculate growth net loss (if enabled, scaled by 1000 for stability)
        if regress_g:
            g_predict = g_net(net_input)  # Predict growth rate
            # Weighted MSE loss between predicted and target growth rate (scaled)
            loss += 1000 * torch.mean(weights * (g_predict - gt_samp) ** 2)

        return loss, penalty

    def evaluate(self, data, time_points):
        """
        Evaluate the trained model by computing Wasserstein-1 distance and TMV (Total Mass Variation)
        between predicted trajectories and real data.
        
        Args:
            data: List of tensors (real samples at each time point).
            time_points: List of time values corresponding to data.
        
        Returns:
            list: Wasserstein-1 distances for each time point (excluding t=0).
        """
        # TODO: this function needs to be improved (handles NaN weights and empty data poorly)

        print(f"\n--- Starting Evaluation ---")
        device = self.device
        x0 = data[0].to(device)  # Initial samples (t=0)
        # Disable gradients for evaluation (save memory and speed up)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get sigma (from training or default to 0.05)
        sigma = getattr(self, 'sigma', None) or 0.05

        # Simulate trajectory using the trained model
        point, weight = simulate_trajectory(
            self.model,
            x0,
            sigma,           
            time_points,
            dt=0.01,  # Time step for ODE simulation
            device=x0.device
        )
        wasserstein_scores = []  # Store Wasserstein distances for each time point
        # Evaluate each time point (excluding t=0)
        for idx in range(1, len(time_points)):
            t0, t1 = time_points[0], time_points[idx]
            data_t1 = data[idx].detach().cpu().numpy()  # Real samples at t1
            x1 = point[idx]  # Predicted samples at t1
            m1 = weight[idx]  # Predicted weights at t1
            # Calculate TMV (Total Mass Variation: deviation from expected mass ratio)
            tmv = np.abs(m1.sum() - data[idx].shape[0]/data[0].shape[0])
            m1 = m1/m1.sum()  # Normalize predicted weights to sum to 1


            # Original (commented) code for Neural ODE step (replaced by simulate_trajectory)
            # x1, lnw1, _ = neural_ode_step(self.ode_func, x0, lnw0, t0, t1, device)
            # m1 = torch.exp(lnw1) / torch.exp(lnw1).sum()

            # Normalize real data weights to sum to 1 (uniform distribution)
            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
            # Compute Euclidean distance matrix between real and predicted samples
            cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

            # Compute Wasserstein-1 distance using EMD (Earth Mover's Distance)
            w1 = ot.emd2(
                m2,
                m1.reshape(-1),  # Flatten weights to 1D
                cost_matrix,
                numItermax=1e7  # Increase max iterations for convergence
            )
            wasserstein_scores.append(w1)
            # Print evaluation results
            print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f}")
            print(f"  Time Point {t1}: TMV = {tmv:.4f}")
        return wasserstein_scores

    def evaluate_stable(self, data, time_points):
        """
        Stable version of the evaluate method: handles NaN weights, empty data, and EMD failures.
        Computes Wasserstein distance (EMD or Sinkhorn fallback) and TMV.
        
        Args:
            data: List of tensors (real samples at each time point).
            time_points: List of time values corresponding to data.
        
        Returns:
            list: Wasserstein distances (EMD/Sinkhorn) for each time point (excluding t=0).
        """
        print(f"\n--- Starting Evaluation ---")
        device = self.device
        x0 = data[0].to(device)  # Initial samples (t=0)
        # Disable gradients for evaluation
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get sigma (from training or default to 0.05)
        sigma = getattr(self, 'sigma', None) or 0.05

        # Simulate trajectory using the trained model
        point, weight = simulate_trajectory(
            self.model,
            x0,
            sigma,           
            time_points,
            dt=0.01,  # Time step for ODE simulation
            device=x0.device
        )
        wasserstein_scores = []  # Store Wasserstein distances
        # Evaluate each time point (excluding t=0)
        for idx in range(1, len(time_points)):
            t0, t1 = time_points[0], time_points[idx]
            data_t1 = data[idx].detach().cpu().numpy()  # Real samples at t1
            x1 = point[idx]  # Predicted samples at t1
            m1 = weight[idx].copy()  # Predicted weights (copy to avoid in-place modification)
            print(m1.shape)
            print(m1.sum())
            print(data[idx].shape[0])
            
            # Detect NaN positions in predicted weights (for debugging)
            nan_positions = np.where(np.isnan(m1))

            # # Print NaN positions (commented out by default)
            # if len(nan_positions[0]) == 0:
            #     print("m1 has no NaN values")
            # else:
            #     print("NaN positions in m1:")
            #     for pos in zip(*nan_positions):
            #         print(pos)
                    
            # Calculate TMV (Total Mass Variation)
            tmv = np.abs(m1.sum() - data[idx].shape[0]/data[0].shape[0])

            # Critical Fix 1: Handle NaN/inf weights and normalize
            m1 = np.nan_to_num(m1, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN/inf with 0
            m1_sum = m1.sum()
            if m1_sum == 0:
                print(f"  Warning: Empty weights at time {t1}, using uniform distribution")
                m1 = np.ones_like(m1) / len(m1)  # Fallback to uniform weights
            else:
                m1 = m1 / m1_sum  # Normalize weights to sum to 1

            # Critical Fix 2: Ensure real and predicted weights have the same total sum (float error correction)
            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
            m2 = m2 * m1.sum()  # Force m2 sum to match m1 sum

            # Critical Fix 3: Skip empty data (avoid errors)
            if data_t1.shape[0] == 0 or x1.shape[0] == 0:
                print(f"  Skip time {t1}: Empty data points")
                wasserstein_scores.append(np.nan)
                continue

            # Compute Euclidean distance matrix between real and predicted samples
            cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

            # Critical Fix 4: Use exception handling for EMD; fallback to Sinkhorn if EMD fails
            try:
                # Compute Wasserstein-1 distance using EMD
                w1 = ot.emd2(
                    m2,
                    m1.reshape(-1),  # Flatten weights to 1D
                    cost_matrix,
                    numItermax=int(1e7)  # Max iterations for EMD
                )
                print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f}")
            except Exception as e:
                # Fallback to Sinkhorn distance (regularized OT) if EMD fails
                print(f"  EMD failed for time {t1} ({str(e)}), using Sinkhorn instead")
                w1 = ot.sinkhorn2(
                    m2,
                    m1.reshape(-1),
                    cost_matrix,
                    reg=0.1,  # Regularization parameter for Sinkhorn
                    numItermax=int(1e7)
                )
                print(f"  Time Point {t1}: Sinkhorn Distance (fallback) = {w1:.4f}")

            print(f"  Time Point {t1}: TMV = {tmv:.4f}")
            wasserstein_scores.append(w1)
            
        return wasserstein_scores
