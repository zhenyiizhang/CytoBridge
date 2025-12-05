import numpy as np, pandas as pd
import torch
import random
import yaml
from copy import deepcopy
from tqdm import tqdm
import ot
import math
from typing import Union

# This block of code is adapted from the torchCFM library
import math
import warnings
from functools import partial
from typing import Optional

import numpy as np
import ot as pot
import torch
import matplotlib.pyplot as plt
from warnings import catch_warnings, simplefilter

# Suppress numerical error warnings from the ot library (specific to line 751 in _sinkhorn.py)
warnings.filterwarnings(
    "ignore",
    message="Numerical errors at iteration.*",  # Match warning content
    category=UserWarning,  # Warning type
    module="ot.unbalanced._sinkhorn"  # Warning source module
)


class OTPlanSampler:
    """
    Sampler for Optimal Transport (OT) plans.
    Supports multiple OT methods to compute transport plans and sample data pairs from them.
    """
    def __init__(
            self,
            method: str,
            reg: float = 0.05,
            reg_m: float = 1.0,
            normalize_cost: bool = False,
            warn: bool = True,
    ) -> None:
        """
        Initialize the OTPlanSampler.
        
        Args:
            method: OT method to use. Options: "exact", "sinkhorn", "unbalanced_sinkhorn", "partial".
            reg: Entropic regularization strength (used for Sinkhorn-based methods).
            reg_m: Mass regularization strength (used for unbalanced Sinkhorn).
            normalize_cost: Whether to normalize the cost matrix by its maximum value.
            warn: Whether to issue warnings for numerical issues in OT plans.
        """
        # Define OT function; expects (a, b, M) as inputs (marginals a/b, cost matrix M)
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced_sinkhorn":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """
        Compute the OT plan (transport matrix) between two datasets x0 (source) and x1 (target).
        
        Args:
            x0: Source dataset tensor, shape [n_samples0, ...].
            x1: Target dataset tensor, shape [n_samples1, ...].
        
        Returns:
            OT plan matrix, shape [n_samples0, n_samples1].
        """
        # Uniform marginals for source and target
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        
        # Flatten high-dimensional data to 2D (required for cost matrix computation)
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        
        # Compute squared Euclidean cost matrix
        M = torch.cdist(x0, x1) ** 2
        
        # Normalize cost matrix if enabled
        if self.normalize_cost:
            M = M / M.max()  # Not recommended for minibatch scenarios
        
        # Compute OT plan (convert to numpy for后续 processing)
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        
        # Check for numerical stability of the OT plan
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        
        # Fallback to uniform plan if OT plan sum is too small (numerical failure)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        
        return p

    def sample_map(self, pi: np.ndarray, batch_size: int, replace: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample indices from the OT plan to get matching pairs between source and target.
        
        Args:
            pi: OT plan matrix, shape [n_samples0, n_samples1].
            batch_size: Number of samples to draw.
            replace: Whether to sample with replacement.
        
        Returns:
            Tuple of (source_indices, target_indices), each shape [batch_size].
        """
        # Flatten OT plan and normalize to get sampling probabilities
        p_flat = pi.flatten()
        p_flat = p_flat / p_flat.sum()
        
        # Sample flat indices and convert back to 2D indices
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p_flat, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0: torch.Tensor, x1: torch.Tensor, replace: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample matching data pairs (x0_i, x1_j) using the OT plan between x0 and x1.
        
        Args:
            x0: Source dataset tensor, shape [n_samples0, ...].
            x1: Target dataset tensor, shape [n_samples1, ...].
            replace: Whether to sample with replacement.
        
        Returns:
            Tuple of (sampled_x0, sampled_x1), each shape [n_samples0, ...].
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_labels(self, x0: torch.Tensor, x1: torch.Tensor, 
                               y0: Optional[torch.Tensor] = None, y1: Optional[torch.Tensor] = None, 
                               replace: bool = True) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample matching data pairs and their corresponding labels using the OT plan.
        
        Args:
            x0: Source dataset tensor, shape [n_samples0, ...].
            x1: Target dataset tensor, shape [n_samples1, ...].
            y0: Labels for source dataset, shape [n_samples0, ...] (optional).
            y1: Labels for target dataset, shape [n_samples1, ...] (optional).
            replace: Whether to sample with replacement.
        
        Returns:
            Tuple of (sampled_x0, sampled_x1, sampled_y0, sampled_y1). 
            Labels are None if not provided.
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X: np.ndarray) -> np.ndarray:
        """
        Sample a continuous trajectory across multiple time steps using OT plans between consecutive steps.
        
        Args:
            X: Time-series dataset, shape [n_samples, n_times, ...].
        
        Returns:
            Sampled trajectory, shape [n_samples, n_times, ...].
        """
        n_times = X.shape[1]
        pis = []
        
        # Compute OT plans between consecutive time steps
        for t in range(n_times - 1):
            pis.append(self.get_map(torch.from_numpy(X[:, t]), torch.from_numpy(X[:, t + 1])))
        
        # Sample indices to form continuous trajectories
        indices = [np.arange(X.shape[0])]
        for pi in pis:
            next_indices = []
            for i in indices[-1]:
                # Sample target index for each source index (using row-wise probabilities)
                next_idx = np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum())
                next_indices.append(next_idx)
            indices.append(np.array(next_indices))
        
        # Extract trajectory data using sampled indices
        trajectory = []
        for t in range(n_times):
            trajectory.append(X[:, t][indices[t]])
        
        return np.stack(trajectory, axis=1)


def wasserstein(
        x0: torch.Tensor,
        x1: torch.Tensor,
        method: Optional[str] = None,
        reg: float = 0.05,
        power: int = 2,
        **kwargs,
) -> float:
    """
    Compute the Wasserstein distance between two datasets x0 and x1.
    
    Args:
        x0: Source dataset tensor, shape [n_samples0, ...].
        x1: Target dataset tensor, shape [n_samples1, ...].
        method: OT method to use. Options: "exact" (default), "sinkhorn".
        reg: Entropic regularization strength (used for Sinkhorn).
        power: Power of the cost function (1 for W1, 2 for W2). Must be 1 or 2.
    
    Returns:
        Computed Wasserstein distance (scalar).
    """
    assert power == 1 or power == 2, "Power must be 1 (W1) or 2 (W2)"
    
    # Define OT function for distance computation
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Uniform marginals
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    
    # Flatten high-dimensional data
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    
    # Compute cost matrix (Euclidean distance, raised to `power` if needed)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M ** 2
    
    # Compute Wasserstein distance (convert to numpy for ot library)
    distance = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    
    # Take square root for W2 (since cost matrix is squared)
    if power == 2:
        distance = math.sqrt(distance)
    
    return distance


def pad_t_like_x(t: Union[float, int, torch.Tensor], x: torch.Tensor) -> Union[float, int, torch.Tensor]:
    """
    Reshape time tensor `t` to match the dimensionality of data tensor `x` (for broadcasting).
    
    Args:
        t: Time value(s) (scalar or tensor).
        x: Data tensor, shape [batch_size, ...].
    
    Returns:
        Reshaped `t` with shape [batch_size, 1, ..., 1] (matches `x`'s batch dim and trailing dims).
    """
    if isinstance(t, (float, int)):
        return t
    # Add trailing singleton dimensions to match x's shape (after batch dim)
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """
    Base class for Conditional Flow Matching (CFM).
    Models the flow between source (x0) and target (x1) distributions at arbitrary time steps t.
    """
    def __init__(self, sigma: Union[float, int] = 0.0):
        """
        Initialize the ConditionalFlowMatcher.
        
        Args:
            sigma: Noise scale for sampling intermediate time-step data (xt).
        """
        self.sigma = sigma

    def compute_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Compute the mean of the intermediate distribution p(xt | x0, x1) at time t.
        Default: Linear interpolation between x0 and x1.
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Time step(s) (scalar or tensor with shape [batch_size]).
        
        Returns:
            Mean tensor mu_t, shape [batch_size, ...].
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """
        Compute the standard deviation of the intermediate distribution p(xt | x0, x1) at time t.
        Default: Constant sigma (independent of t).
        
        Args:
            t: Time step(s) (scalar or tensor).
        
        Returns:
            Standard deviation sigma_t (scalar or tensor).
        """
        del t  # Unused in base class
        return self.sigma

    def sample_xt(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                 epsilon: torch.Tensor) -> torch.Tensor:
        """
        Sample intermediate data xt from p(xt | x0, x1) using reparameterization.
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Time step(s), shape [batch_size].
            epsilon: Noise tensor (from standard normal), shape [batch_size, ...].
        
        Returns:
            Sampled xt tensor, shape [batch_size, ...].
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)  # Match shape for broadcasting
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                                 xt: torch.Tensor) -> torch.Tensor:
        """
        Compute the conditional flow (ut) at intermediate time t and data xt.
        Default: Constant flow (x1 - x0).
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Time step(s), shape [batch_size].
            xt: Intermediate data tensor, shape [batch_size, ...].
        
        Returns:
            Conditional flow ut tensor, shape [batch_size, ...].
        """
        del t, xt  # Unused in base class
        return x1 - x0

    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        """Sample standard normal noise with the same shape as x."""
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, 
                                           t: Optional[torch.Tensor] = None, return_noise: bool = False) -> Union[tuple, tuple]:
        """
        Sample time steps t, intermediate data xt, and corresponding conditional flow ut.
        
        Args:
            x0: Source data tensor, shape [batch_size, ...].
            x1: Target data tensor, shape [batch_size, ...].
            t: Predefined time steps (optional). If None, samples t ~ Uniform(0,1).
            return_noise: Whether to return the noise tensor used to sample xt.
        
        Returns:
            If return_noise: (t, xt, ut, epsilon)
            Else: (t, xt, ut)
        """
        # Sample uniform time steps if not provided
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t must have the same batch size as x0"

        # Sample noise and intermediate data
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        
        # Compute conditional flow
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Compute the lambda term for flow normalization (depends on sigma_t)."""
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma ** 2 + 1e-8)


class ExactOptimalTransportConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    CFM subclass that uses Exact Optimal Transport to sample matching (x0, x1) pairs.
    Ensures pairs are aligned via the exact OT plan before flow computation.
    """
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")

    def sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, 
                                           t: Optional[torch.Tensor] = None, return_noise: bool = False) -> Union[tuple, tuple]:
        # Sample OT-aligned (x0, x1) pairs first
        x0_aligned, x1_aligned = self.ot_sampler.sample_plan(x0, x1)
        # Call parent class method with aligned pairs
        return super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)

    def guided_sample_location_and_conditional_flow(
            self, x0: torch.Tensor, x1: torch.Tensor, y0: Optional[torch.Tensor] = None, 
            y1: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None, return_noise: bool = False
    ) -> Union[tuple, tuple]:
        # Sample OT-aligned (x0, x1) pairs and their labels
        x0_aligned, x1_aligned, y0_aligned, y1_aligned = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)
            return t, xt, ut, y0_aligned, y1_aligned, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)
            return t, xt, ut, y0_aligned, y1_aligned


class TargetConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    CFM subclass where the intermediate distribution p(xt | x1) only depends on the target x1 (not x0).
    Useful for one-sided flow matching scenarios.
    """
    def compute_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Mean of p(xt | x1): Linear interpolation from 0 to x1."""
        del x0  # Unused (only depends on x1)
        t = pad_t_like_x(t, x1)
        return t * x1

    def compute_sigma_t(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Sigma_t: Decreases linearly from 1 to sigma as t goes from 0 to 1."""
        return 1 - (1 - self.sigma) * t

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                                 xt: torch.Tensor) -> torch.Tensor:
        """Conditional flow adjusted for target-only dependence."""
        del x0  # Unused
        t = pad_t_like_x(t, x1)
        return (x1 - (1 - self.sigma) * xt) / (1 - (1 - self.sigma) * t)


class ConditionalRegularizedUnbalancedFlowMatcher(ConditionalFlowMatcher):
    """
    CFM subclass for Regularized Unbalanced OT.
    Incorporates unbalanced OT plans and time-dependent sigma_t for improved flow modeling.
    """
    def __init__(self, sigma: Union[float, int] = 1.0, ot_method: str = "exact"):
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        
        super().__init__(sigma)
        self.ot_method = ot_method
        # Initialize OT sampler with regularization (2*sigma² for entropic term)
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma ** 2)

    def compute_sigma_t(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Sigma_t: Time-dependent (peaks at t=0.5), proportional to sqrt(t*(1-t))."""
        t = pad_t_like_x(t, torch.tensor(0.0))  # Ensure tensor compatibility
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                                 xt: torch.Tensor) -> torch.Tensor:
        """
        Conditional flow for regularized unbalanced OT.
        Combines linear flow (x1 - x0) and a correction term based on xt's deviation from mu_t.
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        
        # Correction term: adjusts flow based on xt's position relative to mu_t
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        
        return ut

    def sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, 
                                           t: Optional[torch.Tensor] = None, uot_plan: Optional[np.ndarray] = None, 
                                           idx_0: Optional[np.ndarray] = None, idx_1: Optional[np.ndarray] = None, 
                                           return_noise: bool = False) -> Union[tuple, tuple]:
        """
        Extended sampling method that incorporates unbalanced OT (UOT) plan information.
        Computes additional terms (gt_samp, weights) for UOT-aware flow matching.
        """
        # Sample time steps if not provided
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t must have the same batch size as x0"

        # Sample noise and intermediate data
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        
        # Compute conditional flow
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        
        # Compute UOT-aware terms (gt_samp: gradient term, weights: mass weights)
        gt_samp, weights = self.compute_cond_g(uot_plan, idx_0, idx_1, t)
        
        if return_noise:
            return t, xt, ut, eps, gt_samp, weights
        else:
            return t, xt, ut, gt_samp, weights

    def compute_cond_g(self, uot_plan: np.ndarray, idx_0: np.ndarray, idx_1: np.ndarray, 
                      t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute UOT-aware gradient term (gt_samp) and mass weights from the UOT plan.
        
        Args:
            uot_plan: Unbalanced OT plan matrix, shape [n_samples0, n_samples1].
            idx_0: Source indices for the current batch.
            idx_1: Target indices for the current batch.
            t: Time steps, shape [batch_size].
        
        Returns:
            Tuple of (gt_samp, weights) – both shape [batch_size, 1].
        """
        # Extract UOT plan rows corresponding to the current batch
        selected_uot_plan = uot_plan[idx_0]
        
        # Compute source-side mass weights (sum over target dimensions)
        source_weights = torch.tensor(
            selected_uot_plan.sum(axis=-1, keepdims=True), 
            dtype=torch.float32,
            device=t.device
        )
        eps = 1e-10  # Avoid log(0) and division by zero
        
        # Compute time-dependent weights (scaled by t)
        weights = (source_weights + eps) ** (t.unsqueeze(1) - 1)
        
        # Compute gradient term (log ratio of source weights to uniform mass)
        gmaps = torch.log(source_weights + eps) - torch.log(
            torch.ones_like(source_weights, device=source_weights.device) + eps
        )
        
        return gmaps, weights

    def guided_sample_location_and_conditional_flow(
            self, x0: torch.Tensor, x1: torch.Tensor, y0: Optional[torch.Tensor] = None, 
            y1: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None, return_noise: bool = False
    ) -> Union[tuple, tuple]:
        # Sample OT-aligned (x0, x1) pairs and their labels
        x0_aligned, x1_aligned, y0_aligned, y1_aligned = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)
            return t, xt, ut, y0_aligned, y1_aligned, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)
            return t, xt, ut, y0_aligned, y1_aligned


class SchrodingerBridgeConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    CFM subclass for Schrodinger Bridge (SB) modeling.
    Uses OT-aligned pairs and time-dependent sigma_t to match SB dynamics.
    """
    def __init__(self, sigma: Union[float, int] = 1.0, ot_method: str = "exact"):
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}.")
        elif sigma < 1e-3:
            warnings.warn("Small sigma values may lead to numerical instability.")
        
        super().__init__(sigma)
        self.ot_method = ot_method
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=2 * self.sigma ** 2)

    def compute_sigma_t(self, t: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Sigma_t: Time-dependent (peaks at t=0.5), proportional to sqrt(t*(1-t))."""
        t = pad_t_like_x(t, torch.tensor(0.0))  # Ensure tensor compatibility
        return self.sigma * torch.sqrt(t * (1 - t))

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                                 xt: torch.Tensor) -> torch.Tensor:
        """
        Conditional flow for Schrodinger Bridge.
        Combines linear flow (x1 - x0) and a correction term based on xt's deviation from mu_t.
        """
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        
        # Correction term: adjusts flow based on xt's position relative to mu_t
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        
        return ut

    def sample_location_and_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, 
                                           t: Optional[torch.Tensor] = None, return_noise: bool = False) -> Union[tuple, tuple]:
        # Sample OT-aligned (x0, x1) pairs first
        x0_aligned, x1_aligned = self.ot_sampler.sample_plan(x0, x1)
        # Call parent class method with aligned pairs
        return super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)

    def guided_sample_location_and_conditional_flow(
            self, x0: torch.Tensor, x1: torch.Tensor, y0: Optional[torch.Tensor] = None, 
            y1: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None, return_noise: bool = False
    ) -> Union[tuple, tuple]:
        # Sample OT-aligned (x0, x1) pairs and their labels
        x0_aligned, x1_aligned, y0_aligned, y1_aligned = self.ot_sampler.sample_plan_with_labels(x0, x1, y0, y1)
        
        if return_noise:
            t, xt, ut, eps = super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)
            return t, xt, ut, y0_aligned, y1_aligned, eps
        else:
            t, xt, ut = super().sample_location_and_conditional_flow(x0_aligned, x1_aligned, t, return_noise)
            return t, xt, ut, y0_aligned, y1_aligned


class VariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    """
    CFM subclass with variance-preserving dynamics.
    Uses trigonometric interpolation (cos/sin) to keep variance constant across time.
    """
    def compute_mu_t(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Mean of p(xt | x0, x1): Trigonometric interpolation (variance-preserving).
        Uses cos(πt/2)*x0 + sin(πt/2)*x1.
        """
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor, t: Union[float, torch.Tensor], 
                                 xt: torch.Tensor) -> torch.Tensor:
        """
        Conditional flow for variance-preserving dynamics.
        Derived from the time derivative of mu_t.
        """
        del xt  # Unused (flow is derived from mu_t's derivative)
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)


def get_batch(FM: ConditionalFlowMatcher, X: torch.Tensor, trajectory: list[torch.Tensor], 
             batch_size: int, n_times: int, return_noise: bool = False) -> Union[tuple, tuple]:
    """
    Construct a training batch by sampling from consecutive time-step pairs in the trajectory.
    
    Args:
        FM: Conditional Flow Matcher instance.
        X: Full dataset tensor (unused in current implementation).
        trajectory: List of tensors, each representing data at a time step (shape [n_samples, ...]).
        batch_size: Batch size (unused; samples match trajectory length).
        n_times: Number of time steps in the trajectory.
        return_noise: Whether to return the noise tensor used to sample xt.
    
    Returns:
        If return_noise: (t, xt, ut, noises)
        Else: (t, xt, ut)
        All tensors are concatenated across time steps.
    """
    ts = []
    xts = []
    uts = []
    noises = []

    # Iterate over consecutive time-step pairs
    for t_start in range(n_times - 1):
        x0 = trajectory[t_start]
        x1 = trajectory[t_start + 1]

        # Sample flow matching terms for the current time pair
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1, return_noise=return_noise)
        
        # Offset time to match global time steps (t_start to t_start + 1)
        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)

    # Concatenate results across all time steps
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    
    return t, xt, ut


def get_batch_size(FM: ConditionalFlowMatcher, X: torch.Tensor, trajectory: list[torch.Tensor], 
                  batch_size: int, time: torch.Tensor, return_noise: bool = False, 
                  hold_one_out: bool = False, hold_out: Optional[int] = None) -> Union[tuple, tuple]:
    """
    Construct a training batch with specified batch size, sampling from non-uniform time steps.
    Adjusts time and flow to account for variable time intervals between consecutive steps.
    
    Args:
        FM: Conditional Flow Matcher instance.
        X: Full dataset tensor (unused in current implementation).
        trajectory: List of tensors, each representing data at a time step (shape [n_samples, ...]).
        batch_size: Number of samples to draw per time step.
        time: Tensor of global time steps (shape [n_times]).
        return_noise: Whether to return the noise tensor used to sample xt.
        hold_one_out: Whether to exclude a time step (unused).
        hold_out: Index of time step to exclude (unused).
    
    Returns:
        If return_noise: (t, xt, ut, noises)
        Else: (t, xt, ut)
        All tensors are concatenated across time steps.
    """
    ts = []
    xts = []
    uts = []
    noises = []

    # Iterate over consecutive time-step pairs (using time indices)
    for idx, t_start in enumerate(time[:-1]):
        x0 = trajectory[idx]
        x1 = trajectory[idx + 1]
        
        # Sample random indices for batch construction
        indices0 = np.random.choice(len(x0), size=batch_size, replace=True)
        indices1 = np.random.choice(len(x1), size=batch_size, replace=True)
        
        # Extract batch data
        x0_batch = x0[indices0]
        x1_batch = x1[indices1]

        # Sample flow matching terms for the current batch
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(x0_batch, x1_batch, return_noise=return_noise)
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(x0_batch, x1_batch, return_noise=return_noise)
        
        # Adjust time to global scale (account for variable time intervals)
        time_interval = time[idx + 1] - time[idx]
        ts.append(t * time_interval + t_start)
        
        # Normalize flow by time interval (to maintain consistent units)
        xts.append(xt)
        uts.append(ut / time_interval)

    # Concatenate results across all time steps
    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)
    
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    
    return t, xt, ut


def calculate_auto_regularization(a: np.ndarray, b: np.ndarray, M: np.ndarray, tolerance: float = 1e-3) -> tuple[float, float]:
    """
    Two-step automatic tuning for unbalanced Sinkhorn regularization parameters:
    Step 1: Select entropic regularization (reg) using the elbow rule on transport cost ⟨G,M⟩.
    Step 2: Select mass regularization (reg_m) via grid search + elbow rule on transport cost.
    
    Args:
        a: Source marginal distribution, shape [n_samples0].
        b: Target marginal distribution, shape [n_samples1].
        M: Cost matrix, shape [n_samples0, n_samples1].
        tolerance: Tolerance for numerical stability checks (unused).
    
    Returns:
        reg: Optimal entropic regularization strength.
        reg_m: Optimal mass regularization strength.
    """
    # Fallback to defaults if cost matrix is empty
    if M.size == 0:
        print("Cost matrix is empty – falling back to default parameters.")
        return 1e-5, 1.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Step 1: Select entropic regularization (reg) via elbow rule
    # ------------------------------------------------------------------
    reg_list, loss_list = [], []  # For logging (unused in final selection)
    fixed_reg_m = 50.0  # Fixed mass regularization for reg selection
    reg = 10.0  # Default fallback value

    # Context manager to convert specific warnings to exceptions (for stability checks)
    class catch_specific_warning:
        def __init__(self, message: str, category: Warning, module: str):
            self.message = message
            self.category = category
            self.module = module

        def __enter__(self):
            self.original_filters = warnings.filters.copy()
            warnings.filterwarnings(
                "error",
                message=self.message,
                category=self.category,
                module=self.module
            )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            warnings.filters = self.original_filters
            return False  # Do not suppress exceptions

    # Check if OT plan is numerically stable (finite values, positive mass, non-zero sum)
    def is_stable(ot_plan: np.ndarray, a: np.ndarray) -> bool:
        return (
            np.all(np.isfinite(ot_plan)) and
            np.sum(ot_plan) > 1e-8 and
            np.all(ot_plan >= 0)
        )

    # Round 1: Coarse search for stable reg (step size = 2e-2)
    eps_min, eps_step, eps_max = 5e-2, 2e-2, 10.0
    current_eps = eps_min
    first_valid_eps = None

    while current_eps <= eps_max:
        try:
            # Catch numerical errors from unbalanced Sinkhorn
            with catch_specific_warning(
                    message="Numerical errors at iteration.*",
                    category=UserWarning,
                    module="ot.unbalanced._sinkhorn"):
                # Compute unbalanced OT plan
                ot_plan = ot.unbalanced.sinkhorn_unbalanced(
                    a=a,
                    b=b,
                    M=M,
                    reg=current_eps,
                    reg_m=[fixed_reg_m, np.inf]
                )
                
                # Check stability of the OT plan
                if is_stable(ot_plan, a):
                    first_valid_eps = current_eps
                    break
        
        except (Exception, UserWarning) as exc:
            print(f"[Round-1 eps={current_eps:.3e}] Failed: {type(exc).__name__}: {exc}")
        
        # Move to next epsilon
        current_eps += eps_step

    # If no valid eps found in coarse search, use default
    if first_valid_eps is None:
        print("No stable eps found in coarse search – keeping default reg =", reg)
    else:
        # Round 2: Fine search for optimal reg (step size = 1e-3)
        fine_eps_min = first_valid_eps + 1e-3
        fine_eps_step = 1e-3
        current_eps = fine_eps_min
        best_eps = None

        while current_eps <= eps_max:
            try:
                with catch_specific_warning(
                        message="Numerical errors at iteration.*",
                        category=UserWarning,
                        module="ot.unbalanced._sinkhorn"):
                    ot_plan = ot.unbalanced.sinkhorn_unbalanced(
                        a=a,
                        b=b,
                        M=M,
                        reg=current_eps,
                        reg_m=[fixed_reg_m, np.inf]
                    )
                    
                    if is_stable(ot_plan, a):
                        best_eps = current_eps
                        break
            
            except (Exception, UserWarning) as exc:
                print(f"[Round-2 eps={current_eps:.3e}] Failed: {type(exc).__name__}: {exc}")
            
            current_eps += fine_eps_step

        # Update reg with fine search result (or coarse search result if fine search fails)
        reg = best_eps if best_eps is not None else first_valid_eps
        print(f"Final entropic reg selected: {reg}")

    # ------------------------------------------------------------------
    # Step 2: Select mass regularization (reg_m) via grid search + elbow rule
    # ------------------------------------------------------------------
    # Log-spaced grid of reg_m candidates (40 points from 1e-2 to 10^1.2)
    reg_m_candidates = np.logspace(-2, 1.2, 40)
    reg_m_list = []
    transport_loss_list = []

    # Evaluate each reg_m candidate
    for reg_m in reg_m_candidates:
        try:
            # Compute unbalanced OT plan with fixed reg (from Step 1)
            G = pot.unbalanced.sinkhorn_unbalanced(
                a=a,
                b=b,
                M=M,
                reg=reg,
                reg_m=[reg_m, np.inf]
            )
            
            # Skip if OT plan is unstable
            if not (np.all(np.isfinite(G)) and G.sum() > 1e-6 and G.min() >= 0):
                continue
            
            # Compute transport cost (⟨G, M⟩)
            transport_loss = float((G * M).sum())
            reg_m_list.append(reg_m)
            transport_loss_list.append(transport_loss)
        
        except Exception:
            continue

    # Handle case with too few valid reg_m candidates
    if len(reg_m_list) < 4:
        # Fallback to reg_m with minimum transport loss
        best_reg_m = reg_m_list[np.argmin(transport_loss_list)] if reg_m_list else 1.0
        print(f"Insufficient valid reg_m candidates – using min-loss reg_m: {best_reg_m}")
    else:
        # Convert to numpy arrays for processing
        x = np.array(reg_m_list, dtype=float)
        y = np.array(transport_loss_list, dtype=float)

        # Normalize x (reg_m) and y (transport loss) to [0, 1]
        x_norm = (x - x[0]) / (x[-1] - x[0] + 1e-12)  # Avoid division by zero
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

        # Define line from first to last point (baseline for elbow detection)
        y0, y1 = y_norm[0], y_norm[-1]
        line_y = y0 + (y1 - y0) * x_norm  # Y-values of the baseline line

        # Compute direction vector of the baseline line
        line_vec = np.array([1.0, y1 - y0])
        line_length = np.linalg.norm(line_vec)

        # Compute perpendicular distance from each point to the baseline
        # Cross product gives perpendicular distance (scaled by line length)
        distances = np.abs(np.cross(
            line_vec, 
            np.column_stack([x_norm - x_norm[0], y_norm - y_norm[0]])
        )) / line_length

        # Elbow is the point with maximum distance to the baseline
        best_reg_m = reg_m_list[np.argmax(distances)]
        print(f"Elbow rule selected reg_m: {best_reg_m:.6f}")

    return reg, best_reg_m


def compute_uot_plans1(X: list[np.ndarray], t_train: torch.Tensor, 
                      use_mini_batch_uot: bool = False, chunk_size: int = 1000) -> tuple[list[np.ndarray], list[Optional[dict]]]:
    """
    Legacy function to compute Unbalanced Optimal Transport (UOT) plans between consecutive time steps.
    Supports full-matrix and mini-batch (chunked) computation.
    
    Args:
        X: List of numpy arrays, each representing data at a time step (shape [n_samples, ...]).
        t_train: Tensor of training time steps (shape [n_times]).
        use_mini_batch_uot: Whether to use mini-batch (chunked) UOT computation.
        chunk_size: Number of samples per chunk (for mini-batch mode).
    
    Returns:
        Tuple of (uot_plans, sampling_info_plans):
            uot_plans: List of UOT plan matrices (each shape [n_samples0, n_samples1]).
            sampling_info_plans: List of sampling info dicts (None for full-matrix mode).
    """
    uot_plans = []
    sampling_info_plans = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate over consecutive time-step pairs
    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):
        X_source, X_target = X[i], X[i + 1]
        n_source, n_target = X_source.shape[0], X_target.shape[0]
        
        # Uniform marginals for source and target
        a = np.ones(n_source)
        b = np.ones(n_target)

        # Compute Euclidean cost matrix
        cost_matrix = pot.dist(X_source, X_target)
        
        # Normalize cost matrix if mean is too large (to improve numerical stability)
        if cost_matrix.mean() > 100:
            cost_matrix = cost_matrix / cost_matrix.max()

        # ------------------------------
        # Full-matrix UOT computation
        # ------------------------------
        if not use_mini_batch_uot:
            # Auto-tune regularization parameters
            reg, reg_m = calculate_auto_regularization(a, b, cost_matrix)
            print(f"Chosen reg and reg_m for time step {i}: {reg}, {reg_m}")
            
            # Convert to CUDA tensors for faster computation
            a_cuda = torch.from_numpy(a).float().to(device)
            b_cuda = torch.from_numpy(b).float().to(device)
            cost_matrix_cuda = torch.from_numpy(cost_matrix).float().to(device)

            # Compute UOT plan
            G = pot.unbalanced.sinkhorn_unbalanced(
                a_cuda, b_cuda, cost_matrix_cuda, reg, [reg_m, np.inf]
            )
            G = G.cpu().numpy()

            # Validate marginal constraints (relaxed check: sum within 1 of target)
            assert (np.abs(G.sum(axis=0) - b) < 1).all(), "UOT plan fails target marginal constraints"
            sampling_info_plans.append(None)

        # ------------------------------
        # Mini-batch UOT computation
        # ------------------------------
        else:
            # Calculate number of chunks (ceil division)
            n_chunks = n_source // chunk_size + 1
            
            # Initialize full UOT plan matrix
            G = np.zeros((n_source, n_target))
            
            # Shuffle source and target indices for chunking
            source_perm = np.arange(n_source)
            np.random.shuffle(source_perm)
            target_perm = np.arange(n_target)
            np.random.shuffle(target_perm)
            
            # Split into chunks
            source_chunks = np.array_split(source_perm, n_chunks)
            target_chunks = np.array_split(target_perm, n_chunks)

            uot_sub_plans = []  # Store UOT plans for individual chunks

            # Process each chunk pair
            for src_chunk, tgt_chunk in zip(source_chunks, target_chunks):
                # Extract chunk-specific cost matrix and marginals
                sub_cost_matrix = cost_matrix[np.ix_(src_chunk, tgt_chunk)]
                sub_a = a[src_chunk]
                sub_b = b[tgt_chunk]

                # Auto-tune regularization using the first chunk
                if len(uot_sub_plans) == 0:
                    reg, reg_m = calculate_auto_regularization(sub_a, sub_b, sub_cost_matrix)
                    print(f"Chosen reg and reg_m (mini-batch) for time step {i}: {reg}, {reg_m}")

                # Convert chunk data to CUDA
                sub_a_cuda = torch.from_numpy(sub_a).float().to(device)
                sub_b_cuda = torch.from_numpy(sub_b).float().to(device)
                sub_cost_matrix_cuda = torch.from_numpy(sub_cost_matrix).float().to(device)

                # Compute UOT plan for the chunk
                G_sub = pot.unbalanced.sinkhorn_unbalanced(
                    sub_a_cuda, sub_b_cuda, sub_cost_matrix_cuda, reg, [reg_m, np.inf]
                )
                G_sub = G_sub.cpu().numpy()

                # Assign chunk plan to full UOT plan matrix
                G[np.ix_(src_chunk, tgt_chunk)] = G_sub

                # Validate chunk marginal constraints (relaxed for mini-batch)
                assert (np.abs(G_sub.sum(axis=0) - sub_b) < 0.1 * chunk_size).all(), \
                    "Chunk UOT plan fails target marginal constraints"
                
                uot_sub_plans.append(G_sub.astype(np.float32))

            # Store sampling info for mini-batch mode (chunk indices + sub-plans)
            sampling_info = {
                'sub_plans': uot_sub_plans,
                'source_groups': source_chunks,
                'target_groups': target_chunks
            }
            sampling_info_plans.append(sampling_info)

        uot_plans.append(G)

    return uot_plans, sampling_info_plans


def compute_uot_plans(
    X: list[np.ndarray], 
    t_train: torch.Tensor, 
    use_mini_batch_uot: bool = False, 
    chunk_size: int = 1000,
    alpha_regm: float = 1.0,
    reg_strategy: str = "max_over_time"  # Options: "per_time" (per time-step) or "max_over_time" (global max)
) -> tuple[list[np.ndarray], list[Optional[dict]]]:
    """
    Compute Unbalanced Optimal Transport (UOT) plans between consecutive time steps.
    Supports two regularization strategies and mini-batch (chunked) computation.
    
    Args:
        X: List of numpy arrays, each representing data at a time step (shape [n_samples, ...]).
        t_train: Tensor of training time steps (shape [n_times]).
        use_mini_batch_uot: Whether to use mini-batch (chunked) UOT computation.
        chunk_size: Number of samples per chunk (for mini-batch mode).
        alpha_regm: Scaling factor for mass regularization (reg_m = alpha_regm * auto_tuned_reg_m).
        reg_strategy: Strategy for selecting regularization parameters:
            "per_time": Auto-tune reg/reg_m independently for each time step.
            "max_over_time": Auto-tune for all time steps, then use global max reg/reg_m.
    
    Returns:
        Tuple of (uot_plans, sampling_info_plans):
            uot_plans: List of UOT plan matrices (each shape [n_samples0, n_samples1]).
            sampling_info_plans: List of sampling info dicts (None for full-matrix mode).
    """
    # Validate regularization strategy
    if reg_strategy not in ["per_time", "max_over_time"]:
        raise ValueError(f"reg_strategy must be 'per_time' or 'max_over_time', got {reg_strategy}")
    
    uot_plans = []
    sampling_info_plans = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_reg = None  # Global entropic reg (for "max_over_time" strategy)
    global_reg_m = None  # Global mass reg (for "max_over_time" strategy)

    # ------------------------------------------------------------------
    # Precompute global regularization parameters (for "max_over_time" strategy)
    # ------------------------------------------------------------------
    if reg_strategy == "max_over_time":
        reg_list = []
        reg_m_list = []
        print("Precomputing regularization parameters for all time steps...")
        
        # Auto-tune reg/reg_m for each time step to find global max
        for i in range(len(t_train) - 1):
            X_source, X_target = X[i], X[i + 1]
            n_source, n_target = X_source.shape[0], X_target.shape[0]
            a = np.ones(n_source)
            b = np.ones(n_target)

            # Compute and normalize cost matrix
            cost_matrix = pot.dist(X_source, X_target)
            if cost_matrix.mean() > 100:
                cost_matrix = cost_matrix / cost_matrix.max()

            # Use chunk data for mini-batch mode (match actual computation logic)
            if use_mini_batch_uot:
                # Shuffle and split into chunks (same as mini-batch computation)
                n_chunks = n_source // chunk_size + 1
                source_perm = np.arange(n_source)
                np.random.shuffle(source_perm)
                target_perm = np.arange(n_target)
                np.random.shuffle(target_perm)
                source_chunks = np.array_split(source_perm, n_chunks)
                target_chunks = np.array_split(target_perm, n_chunks)
                
                # Use first chunk to compute reg/reg_m (consistent with mini-batch mode)
                first_src_chunk = source_chunks[0]
                first_tgt_chunk = target_chunks[0]
                sub_cost_matrix = cost_matrix[np.ix_(first_src_chunk, first_tgt_chunk)]
                sub_a = a[first_src_chunk]
                sub_b = b[first_tgt_chunk]
                
                reg, reg_m = calculate_auto_regularization(sub_a, sub_b, sub_cost_matrix)
                print(f"Time step {i} (mini-batch first chunk): reg={reg}, reg_m={reg_m}")
            else:
                # Use full matrix for reg/reg_m computation
                reg, reg_m = calculate_auto_regularization(a, b, cost_matrix)
                print(f"Time step {i} (full matrix): reg={reg}, reg_m={reg_m}")

            # Collect reg/reg_m for global max calculation
            reg_list.append(reg)
            reg_m_list.append(reg_m)

        # Set global parameters to the maximum of all time-step values
        global_reg = max(reg_list)
        global_reg_m = max(reg_m_list)
        print(f"Global max regularization parameters: reg={global_reg}, reg_m={global_reg_m}")

    # ------------------------------------------------------------------
    # Compute UOT plans for each time-step pair
    # ------------------------------------------------------------------
    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):
        X_source, X_target = X[i], X[i + 1]
        n_source, n_target = X_source.shape[0], X_target.shape[0]
        a = np.ones(n_source)
        b = np.ones(n_target)

        # Compute and normalize cost matrix
        cost_matrix = pot.dist(X_source, X_target)
        if cost_matrix.mean() > 100:
            cost_matrix = cost_matrix / cost_matrix.max()

        # Determine regularization parameters for current time step
        if reg_strategy == "per_time":
            # Auto-tune independently for each time step
            if not use_mini_batch_uot:
                reg, reg_m = calculate_auto_regularization(a, b, cost_matrix)
                print(f"Time step {i} (per_time strategy): reg={reg}, reg_m={reg_m}")
            else:
                # Mini-batch mode: auto-tune later (using first chunk)
                reg, reg_m = None, None
        else:
            # Use precomputed global max parameters
            reg, reg_m = global_reg, global_reg_m
            print(f"Time step {i} (max_over_time strategy): using reg={reg}, reg_m={reg_m}")

        # ------------------------------
        # Full-matrix UOT computation
        # ------------------------------
        if not use_mini_batch_uot:
            # Scale mass regularization with alpha_regm
            scaled_reg_m = reg_m * alpha_regm
            print(f"Time step {i}: scaled reg_m = {scaled_reg_m}")
            
            # Convert to CUDA for faster computation
            a_cuda = torch.from_numpy(a).float().to(device)
            b_cuda = torch.from_numpy(b).float().to(device)
            cost_matrix_cuda = torch.from_numpy(cost_matrix).float().to(device)

            # Compute UOT plan
            G = pot.unbalanced.sinkhorn_unbalanced(
                a_cuda, b_cuda, cost_matrix_cuda, reg, [scaled_reg_m, np.inf]
            )
            G = G.cpu().numpy()

            # Validate marginal constraints
            assert (np.abs(G.sum(axis=0) - b) < 1).all(), "UOT plan fails target marginal constraints"
            sampling_info_plans.append(None)

        # ------------------------------
        # Mini-batch UOT computation
        # ------------------------------
        else:
            # Calculate number of chunks
            n_chunks = n_source // chunk_size + 1
            
            # Initialize full UOT plan matrix
            G = np.zeros((n_source, n_target))
            
            # Shuffle and split indices into chunks
            source_perm = np.arange(n_source)
            np.random.shuffle(source_perm)
            target_perm = np.arange(n_target)
            np.random.shuffle(target_perm)
            source_chunks = np.array_split(source_perm, n_chunks)
            target_chunks = np.array_split(target_perm, n_chunks)

            uot_sub_plans = []  # Store chunk-specific UOT plans

            # Process each chunk pair
            for src_chunk, tgt_chunk in zip(source_chunks, target_chunks):
                # Extract chunk data
                sub_cost_matrix = cost_matrix[np.ix_(src_chunk, tgt_chunk)]
                sub_a = a[src_chunk]
                sub_b = b[tgt_chunk]

                # Auto-tune reg/reg_m using the first chunk (for "per_time" strategy)
                if reg_strategy == "per_time" and len(uot_sub_plans) == 0:
                    reg, reg_m = calculate_auto_regularization(sub_a, sub_b, sub_cost_matrix)
                    print(f"Time step {i} chunk 0 (per_time): reg={reg}, reg_m={reg_m}")

                # Scale mass regularization
                scaled_reg_m = reg_m * alpha_regm
                print(f"Time step {i} chunk: scaled reg_m = {scaled_reg_m}")

                # Convert chunk data to CUDA
                sub_a_cuda = torch.from_numpy(sub_a).float().to(device)
                sub_b_cuda = torch.from_numpy(sub_b).float().to(device)
                sub_cost_matrix_cuda = torch.from_numpy(sub_cost_matrix).float().to(device)

                # Compute UOT plan for the chunk
                G_sub = pot.unbalanced.sinkhorn_unbalanced(
                    sub_a_cuda, sub_b_cuda, sub_cost_matrix_cuda, reg, [scaled_reg_m, np.inf]
                )
                G_sub = G_sub.cpu().numpy()

                # Assign chunk plan to full matrix
                G[np.ix_(src_chunk, tgt_chunk)] = G_sub

                # Validate chunk marginal constraints
                assert (np.abs(G_sub.sum(axis=0) - sub_b) < 0.1 * chunk_size).all(), \
                    "Chunk UOT plan fails target marginal constraints"
                
                uot_sub_plans.append(G_sub.astype(np.float32))

            # Store sampling info for mini-batch mode
            sampling_info = {
                'sub_plans': uot_sub_plans,
                'source_groups': source_chunks,
                'target_groups': target_chunks
            }
            sampling_info_plans.append(sampling_info)

        uot_plans.append(G)

    return uot_plans, sampling_info_plans


def sample_from_ot_plan(
        ot_plan: np.ndarray,
        x0: np.ndarray,
        x1: np.ndarray,
        batch_size: int,
        sampling_info: Optional[dict] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample matching (x0, x1) pairs from an OT plan.
    Supports both full-matrix and mini-batch (chunked) OT plans.
    
    Args:
        ot_plan: OT plan matrix (full: [n_samples0, n_samples1]; chunked: aggregated from sub-plans).
        x0: Source dataset array, shape [n_samples0, ...].
        x1: Target dataset array, shape [n_samples1, ...].
        batch_size: Number of pairs to sample.
        sampling_info: Sampling info dict for mini-batch mode (contains chunk indices and sub-plans).
    
    Returns:
        Tuple of (sampled_x0, sampled_x1, sampled_i, sampled_j):
            sampled_x0: Sampled source data, shape [batch_size, ...].
            sampled_x1: Sampled target data, shape [batch_size, ...].
            sampled_i: Source indices of sampled pairs, shape [batch_size].
            sampled_j: Target indices of sampled pairs, shape [batch_size].
            Empty arrays if OT plan has insufficient mass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Sampling from full OT plan
    # ------------------------------
    if sampling_info is None:
        # Convert OT plan to CUDA tensor
        pi_cuda = torch.from_numpy(ot_plan.astype(np.float32)).to(device)
        
        # Compute row sums (source-side mass) and total mass
        row_sums = pi_cuda.sum(axis=1)
        total_mass = row_sums.sum()
        
        # Return empty if total mass is too small (numerical failure)
        if total_mass < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Sample source indices (proportional to row sums)
        row_probs = row_sums / total_mass
        sampled_i = torch.multinomial(row_probs, num_samples=batch_size, replacement=True)
        
        # Sample target indices (conditional on source indices)
        selected_rows = pi_cuda[sampled_i]
        selected_row_sums = row_sums[sampled_i]
        conditional_probs = selected_rows / (selected_row_sums.unsqueeze(1) + 1e-12)  # Avoid division by zero
        sampled_j = torch.multinomial(conditional_probs, num_samples=1).squeeze(1)
        
        # Convert indices to numpy
        sampled_i_np = sampled_i.cpu().numpy()
        sampled_j_np = sampled_j.cpu().numpy()

    # ------------------------------
    # Sampling from mini-batch OT plan
    # ------------------------------
    else:
        # Extract chunk info from sampling_info
        sub_plans = sampling_info['sub_plans']
        source_chunks = sampling_info['source_groups']
        target_chunks = sampling_info['target_groups']
        
        # Compute chunk masses and total mass
        chunk_masses = [sub_plan.sum() for sub_plan in sub_plans]
        total_mass = sum(chunk_masses)
        
        # Return empty if total mass is too small
        if total_mass < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Sample chunks (proportional to chunk masses)
        chunk_probs = torch.tensor(chunk_masses, dtype=torch.float32, device=device) / total_mass
        sampled_chunk_indices = torch.multinomial(chunk_probs, num_samples=batch_size, replacement=True)
        
        # Convert chunk data to CUDA for faster processing
        sub_plans_cuda = [torch.from_numpy(sp).to(device) for sp in sub_plans]
        source_chunks_cuda = [torch.from_numpy(sc).to(device) for sc in source_chunks]
        target_chunks_cuda = [torch.from_numpy(tc).to(device) for tc in target_chunks]
        
        # Get unique chunks and their sample counts
        unique_chunks, chunk_counts = torch.unique(sampled_chunk_indices, return_counts=True)
        
        # Initialize arrays for sampled indices
        sampled_i_cuda = torch.empty(batch_size, dtype=torch.long, device=device)
        sampled_j_cuda = torch.empty(batch_size, dtype=torch.long, device=device)

        # Process each unique chunk
        for chunk_idx, count in zip(unique_chunks, chunk_counts):
            # Get current chunk's OT plan and indices
            chunk_plan = sub_plans_cuda[chunk_idx]
            chunk_source_idx = source_chunks_cuda[chunk_idx]
            chunk_target_idx = target_chunks_cuda[chunk_idx]
            
            # Compute row sums (source-side mass) for the chunk
            chunk_row_sums = chunk_plan.sum(axis=1)
            chunk_total_mass = chunk_row_sums.sum()
            
            # Skip if chunk has insufficient mass
            if chunk_total_mass < 1e-9:
                continue
            
            # Sample source indices within the chunk
            chunk_row_probs = chunk_row_sums / chunk_total_mass
            local_source_idx = torch.multinomial(chunk_row_probs, num_samples=count.item(), replacement=True)
            
            # Sample target indices within the chunk (conditional on source)
            selected_chunk_rows = chunk_plan[local_source_idx]
            selected_chunk_row_sums = chunk_row_sums[local_source_idx]
            chunk_conditional_probs = selected_chunk_rows / (selected_chunk_row_sums.unsqueeze(1) + 1e-12)
            local_target_idx = torch.multinomial(chunk_conditional_probs, num_samples=1).squeeze(1)
            
            # Map local chunk indices to global dataset indices
            global_source_idx = chunk_source_idx[local_source_idx]
            global_target_idx = chunk_target_idx[local_target_idx]
            
            # Assign to final index arrays (using mask for chunk position)
            chunk_mask = (sampled_chunk_indices == chunk_idx)
            sampled_i_cuda[chunk_mask] = global_source_idx.long()
            sampled_j_cuda[chunk_mask] = global_target_idx.long()
        
        # Convert CUDA indices to numpy
        sampled_i_np = sampled_i_cuda.cpu().numpy()
        sampled_j_np = sampled_j_cuda.cpu().numpy()

    # Return empty arrays if no valid indices were sampled
    if sampled_i_np.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Extract sampled data pairs using indices
    sampled_x0 = x0[sampled_i_np]
    sampled_x1 = x1[sampled_j_np]
    
    return sampled_x0, sampled_x1, sampled_i_np, sampled_j_np


def get_batch_uot_fm(FM: ConditionalRegularizedUnbalancedFlowMatcher, X: list[np.ndarray], 
                    t_train: torch.Tensor, batch_size: int, uot_plans: list[np.ndarray], 
                    sampling_info_plans: list[Optional[dict]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct a training batch for Unbalanced Optimal Transport (UOT)-aware Conditional Flow Matching (CFM).
    Samples aligned (x0, x1) pairs from UOT plans and computes flow matching terms.
    
    Args:
        FM: UOT-aware CFM instance (ConditionalRegularizedUnbalancedFlowMatcher).
        X: List of numpy arrays, each representing data at a time step (shape [n_samples, ...]).
        t_train: Tensor of training time steps (shape [n_times]).
        batch_size: Number of samples to draw per time step.
        uot_plans: List of UOT plan matrices (each shape [n_samples0, n_samples1] for consecutive time steps).
        sampling_info_plans: List of sampling info dicts (None for full-matrix UOT plans).
    
    Returns:
        Tuple of concatenated tensors across all time steps:
            t: Time steps (shape [total_samples]).
            xt: Intermediate data (shape [total_samples, ...]).
            ut: Conditional flow (shape [total_samples, ...]).
            gt: UOT-aware gradient term (shape [total_samples, 1]).
            mass_t: UOT mass weights (shape [total_samples, 1]).
            noises: Noise used to sample xt (shape [total_samples, ...]).
    """
    ts = []
    xts = []
    uts = []
    gts = []
    mass_ts = []
    noises = []

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Iterate over consecutive time-step pairs (match UOT plans to time steps)
    for time_idx in range(t_train.size(0) - 1):
        # Get UOT plan and sampling info for current time pair
        current_uot_plan = uot_plans[time_idx]
        current_sampling_info = sampling_info_plans[time_idx]
        
        # Get source (t) and target (t+1) data for current time pair
        x0_np = X[time_idx]
        x1_np = X[time_idx + 1]
        
        # Sample aligned (x0, x1) pairs from UOT plan
        sampled_x0_np, sampled_x1_np, sampled_idx0, sampled_idx1 = sample_from_ot_plan(
            ot_plan=current_uot_plan,
            x0=x0_np,
            x1=x1_np,
            batch_size=batch_size,
            sampling_info=current_sampling_info
        )
        
        # Skip if sampling failed (no valid pairs)
        if sampled_x0_np.size == 0:
            continue
        
        # Convert sampled data to CUDA tensors
        sampled_x0 = torch.from_numpy(sampled_x0_np).float().to(device)
        sampled_x1 = torch.from_numpy(sampled_x1_np).float().to(device)
        
        # Compute time interval between current and next time step
        time_interval = t_train[time_idx + 1] - t_train[time_idx]
        
        # Sample flow matching terms (includes UOT-aware gt and mass weights)
        t_sampled, xt, ut, eps, gt_samp, mass_weights = FM.sample_location_and_conditional_flow(
            x0=sampled_x0,
            x1=sampled_x1,
            uot_plan=current_uot_plan,
            idx_0=sampled_idx0,
            idx_1=sampled_idx1,
            return_noise=True
        )
        
        # Adjust time to global scale (offset by current time step)
        ts.append(t_sampled * time_interval + t_train[time_idx])
        
        # Adjust flow to account for time interval (normalize to per-unit time)
        xts.append(xt)
        uts.append(ut / time_interval)
        
        # Adjust UOT-aware terms to per-unit time
        gts.append(gt_samp / time_interval)
        mass_ts.append(mass_weights)
        
        # Store noise (no adjustment needed)
        noises.append(eps)

    # Concatenate all tensors across time steps (handle empty case)
    concat_t = torch.cat(ts) if ts else torch.empty(0, device=device)
    concat_xt = torch.cat(xts) if xts else torch.empty(0, *X[0].shape[1:], device=device)
    concat_ut = torch.cat(uts) if uts else torch.empty(0, *X[0].shape[1:], device=device)
    concat_gt = torch.cat(gts) if gts else torch.empty(0, 1, device=device)
    concat_mass_t = torch.cat(mass_ts) if mass_ts else torch.empty(0, 1, device=device)
    concat_noises = torch.cat(noises) if noises else torch.empty(0, *X[0].shape[1:], device=device)

    return concat_t, concat_xt, concat_ut, concat_gt, concat_mass_t, concat_noises
