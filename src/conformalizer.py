# src/conformalizer.py

import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from scipy.optimize import bisect
from tqdm import tqdm
import torch.utils.data as tdata
from numba import jit
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from src.conformalizer_old import NumbaConvexHullMethod
from .utils import get_logits_targets, platt_logits, platt_scale

# ============================================================================
# Abstract Base Classes
# ============================================================================

class PredictionSetMethod(ABC):
    """Abstract base class for different conformal prediction methods."""
    
    def __init__(self, alpha: float):
        self.alpha = alpha
    
    @abstractmethod
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        """Compute conformity score for calibration."""
        pass
    
    @abstractmethod
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        """Compute prediction set given probabilities and threshold."""
        pass
    
    @abstractmethod
    def is_conformity_score(self) -> bool:
        """Return True if this method uses conformity scores (higher = more conforming).
        Return False if this method uses non-conformity scores (higher = less conforming)."""
        pass
    
    def needs_calibration(self) -> bool:
        """Return True if this method needs conformity score calibration."""
        return True


# ============================================================================
# Naive Methods
# ============================================================================

class Naive(PredictionSetMethod):
    """
    Adaptive Prediction Sets (APS) method without calibration.
    """
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        raise NotImplementedError("APS method doesn't use conformity scores")
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float = None) -> np.ndarray:
        p_hat_sorted_indices = np.argsort(p_hat)[::-1]
        p_hat_sorted = p_hat[p_hat_sorted_indices]
        p_hat_cumulative = np.cumsum(p_hat_sorted)
        try:
            k = np.where(p_hat_cumulative > (1 - self.alpha))[0][0]
            prediction_set = p_hat_sorted_indices[:k+1]
        except IndexError:
            prediction_set = p_hat_sorted_indices
        return prediction_set
    
    
    def is_conformity_score(self) -> bool:
        raise NotImplementedError("is_conformity_score is not applicable to the Naive method.")
    
    def needs_calibration(self) -> bool:
        return False


# ============================================================================
# Regularized Adaptive Prediction Sets (RAPS)
# ============================================================================

class RAPS(PredictionSetMethod):
    """
    Regularized Adaptive Prediction Sets (RAPS) method.
    """
    def __init__(self, alpha: float, k_reg: int = 1, penalty_lambda: float = 0.01, randomized: bool = False):
        super().__init__(alpha)
        if k_reg < 0:
            raise ValueError("k_reg must be non-negative.")
        if penalty_lambda < 0:
            raise ValueError("penalty_lambda must be non-negative.")
        self.k_reg = k_reg
        self.penalty_lambda = penalty_lambda
        self.randomized = randomized

    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        """
        Compute non-conformity score for RAPS.
        """
        p_hat_sorted_indices = np.argsort(p_hat)[::-1]
        p_hat_sorted = p_hat[p_hat_sorted_indices]
        
        try:
            l_i = np.where(p_hat_sorted_indices == target)[0][0] + 1
        except IndexError:
            return np.inf

        cumulative_prob = np.sum(p_hat_sorted[:l_i])
        regularization = self.penalty_lambda * max(0, l_i - self.k_reg)
        e_i = cumulative_prob + regularization
        
        if self.randomized:
            u = np.random.uniform(0, 1)
            s_li = p_hat_sorted[l_i - 1]
            e_i = e_i - s_li + u * s_li
            
        return e_i

    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        """
        Compute the RAPS prediction set.
        """
        K = p_hat.shape[0]
        lam = float(self.penalty_lambda)
        kreg = int(self.k_reg)

        order = np.argsort(p_hat)[::-1]  
        s = p_hat[order] 
        csum = np.cumsum(s)
        j = np.arange(1, K + 1, dtype=float)
        penalty = lam * np.maximum(0.0, j - kreg)
        S = csum + penalty  

        L = int(np.sum(S <= threshold)) + 1

        if L > K:
            return order 

        if self.randomized and L >= 1:
            sL = s[L-1]
            V = (threshold - (S[L-1] - sL)) / sL
            
            U = np.random.uniform(0.0, 1.0)
            if V <= U:
                L = max(0, L - 1)  

        if L <= 0:
            return np.array([], dtype=order.dtype)
        return order[:L]

    def is_conformity_score(self) -> bool:
        return False


# ============================================================================
# Special Cases: (Lambda = 0 or infinity)
# ============================================================================

class PureSingleton(PredictionSetMethod):
    """
    M-Criterion/Pure Singleton method, a special case for penalty_lambda = 0.
    """
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        """
        Compute non-conformity score for the M-Criterion method.
        """
        sorted_indices = np.argsort(p_hat)[::-1]
        p_sorted = p_hat[sorted_indices]
        try:
            i = np.where(sorted_indices == target)[0][0] + 1
        except IndexError:
            return np.inf

        if i == 1:
            return 0.0
        else:
            p1 = p_sorted[0]
            if p1 >= 1.0 - 1e-9: 
                return np.inf
            return 1.0 / (1.0 - p1)

    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        """
        Compute prediction set for the M-Criterion method.
        """
        sorted_indices = np.argsort(p_hat)[::-1]
        p1 = p_hat[sorted_indices[0]]

        if p1 >= 1.0 - 1e-9:
            return sorted_indices[:1]

        r_others = 1.0 / (1.0 - p1)

        if r_others <= threshold:
            return sorted_indices
        else:
            return sorted_indices[:1]


    def is_conformity_score(self) -> bool:
        return False


class LAS(PredictionSetMethod):
    """
    Least Ambiguous Sets method, a special case for Lambda = infinity.
    """
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        return p_hat[target]
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(p_hat >= threshold)[0]
    
    
    def is_conformity_score(self) -> bool:
        return True
    

# ============================================================================
# Singleton-Optimized Conformal Prediction Method
# ============================================================================

class SOCOP(PredictionSetMethod):
    """
    Singleton-Optimized Conformal Prediction using the geometric Convex Hull approach.
    """
    def __init__(self, alpha: float, penalty_lambda: float = 0.2, use_numba: bool = False):
        super().__init__(alpha)
        if penalty_lambda <= 0:
            raise ValueError("penalty_lambda must be positive for SOCOP.")
        self.penalty_lambda = penalty_lambda
        self.use_numba = use_numba

        if self.use_numba:
            # Warm-up call to compile the function
            _ = _compute_lower_convex_hull_numba(np.array([0.5, 0.3, 0.2]), self.penalty_lambda)

    def _compute_lower_convex_hull(self, p_hat_sorted: np.ndarray):
        """
        Computes the lower convex hull of the point set P_k = (S_k, g_k).
        """
        if self.use_numba:
            return _compute_lower_convex_hull_numba(p_hat_sorted, self.penalty_lambda)
        else:
            K = len(p_hat_sorted)
            
            S = np.zeros(K + 1)
            S[1:] = np.cumsum(p_hat_sorted)
            g = (np.arange(K + 1) > 1).astype(float) + self.penalty_lambda * np.arange(K + 1)
            
            lower_hull_indices = []
            for k in range(K + 1):
                while len(lower_hull_indices) >= 2:
                    i = lower_hull_indices[-1]
                    j = lower_hull_indices[-2]
                    cross_product = (S[i] - S[j]) * (g[k] - g[i]) - (S[k] - S[i]) * (g[i] - g[j])
                    
                    if cross_product <= 1e-9:
                        lower_hull_indices.pop()
                    else:
                        break 
                lower_hull_indices.append(k)
                
            return np.array(lower_hull_indices), S, g

    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        """
        Compute non-conformity score for SOCOP.
        """
        sorted_indices = np.argsort(p_hat)[::-1]
        p_sorted = p_hat[sorted_indices]
        try:
            i = np.where(sorted_indices == target)[0][0] + 1
        except IndexError:
            return np.inf

        hull_indices, S, g = self._compute_lower_convex_hull(p_sorted)

        found_j = -1
        for j, v_j in enumerate(hull_indices):
            if v_j >= i:
                found_j = j
                break
        
        if found_j <= 0: 
             if len(hull_indices) > 1 and hull_indices[0] < i <= hull_indices[1]:
                 found_j = 1
             else: 
                 return 0.0

        v_j = hull_indices[found_j]
        v_prev = hull_indices[found_j - 1]
        
        delta_S = S[v_j] - S[v_prev]
        
        return np.inf if delta_S < 1e-12 else (g[v_j] - g[v_prev]) / delta_S

    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        """
        Compute the prediction set for SOCOP.
        """
        sorted_indices = np.argsort(p_hat)[::-1]
        p_sorted = p_hat[sorted_indices]

        hull_indices, S, g = self._compute_lower_convex_hull(p_sorted)
        k_final = hull_indices[0] 
        for j in range(1, len(hull_indices)):
            v_prev = hull_indices[j - 1]
            v_curr = hull_indices[j]
            
            delta_S = S[v_curr] - S[v_prev]
            slope = np.inf if delta_S < 1e-12 else (g[v_curr] - g[v_prev]) / delta_S

            if slope <= threshold:
                k_final = v_curr
            else:
                break
        
        return sorted_indices[:k_final]
    
    
    def is_conformity_score(self) -> bool:
        return False  


# ==============================================================================
# NUMBA JIT-ACCELERATED CONVEX HULL METHOD
# ==============================================================================

@jit(nopython=True, cache=True)
def _compute_lower_convex_hull_numba(p_hat_sorted: np.ndarray, penalty_lambda: float):
    K = len(p_hat_sorted)
    S = np.zeros(K + 1, dtype=np.float64)
    g = np.zeros(K + 1, dtype=np.float64)
    
    current_S = 0.0
    for k in range(K):
        current_S += p_hat_sorted[k]
        S[k+1] = current_S
    for k in range(K + 1):
        g[k] = (1.0 if k > 1 else 0.0) + penalty_lambda * k

    lower_hull_indices_arr = np.zeros(K + 1, dtype=np.int64)
    hull_size = 0
    for k in range(K + 1):
        while hull_size >= 2:
            i = lower_hull_indices_arr[hull_size - 1]
            j = lower_hull_indices_arr[hull_size - 2]
            cross_product = (S[i] - S[j]) * (g[k] - g[i]) - (S[k] - S[i]) * (g[i] - g[j])
            if cross_product <= 1e-9:
                hull_size -= 1
            else:
                break
        lower_hull_indices_arr[hull_size] = k
        hull_size += 1
    return lower_hull_indices_arr[:hull_size], S, g


# ============================================================================
# Factory Pattern for Method Creation
# ============================================================================

class MethodFactory:
    """Factory class for creating prediction set methods."""
    
    _methods = {
        'naive': Naive,
        'raps': RAPS,
        'pure-singleton': PureSingleton,
        'las': LAS,
        'socop': SOCOP,
    }
    
    @classmethod
    def create_method(cls, method_name: str, alpha: float, **kwargs) -> PredictionSetMethod:
        """
        Create a prediction set method instance.
        
        Args:
            method_name: Name of the method to create
            alpha: Miscoverage level
            **kwargs: Additional method-specific parameters
        """
        if method_name not in cls._methods:
            available = list(cls._methods.keys())
            raise ValueError(f"Unknown method '{method_name}'. Available methods: {available}")
        
        method_class = cls._methods[method_name]
        
        if method_name == 'raps':
            return method_class(
                alpha,
                k_reg=kwargs.get('k_reg', 1),
                penalty_lambda=kwargs.get('penalty_lambda', 0.01),
                randomized=kwargs.get('randomized', False)
            )
        elif method_name == 'socop':
            return method_class(
                alpha, 
                penalty_lambda=kwargs.get('penalty_lambda', 0.2),
                use_numba=kwargs.get('use_numba', False)
            )
        else:
            return method_class(alpha)
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """Get list of available method names."""
        return list(cls._methods.keys())


# ============================================================================
# Main Conformal Predictor
# ============================================================================

class ConformalPredictor:
    """
    Low-level conformal predictor that operates on logits only.
    This provides a clean separation between model inference and conformal prediction.
    
    Args:
        calib_logits: TensorDataset containing calibration logits and targets
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        method: Conformal prediction method name
        temperature: Temperature parameter for scaling logits (default: 1.0)
        verbose: Whether to print progress information
    """
    
    def __init__(
        self, 
        calib_logits: tdata.TensorDataset,
        alpha: float = 0.05, 
        method: str = 'pure-singleton',
        temperature: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        self.alpha = alpha
        self.method_name = method
        self.temperature = temperature
        self.verbose = verbose
        
        if not isinstance(calib_logits, tdata.TensorDataset):
            raise ValueError("calib_logits must be a TensorDataset (use get_logits_targets to create it)")
        
        calib_logits_tensor = calib_logits.tensors[0]
        calib_targets_tensor = calib_logits.tensors[1]
        
        self.prediction_method = MethodFactory.create_method(method, alpha, **kwargs)
        self._setup_calibration(calib_logits_tensor, calib_targets_tensor)
    
    def _setup_calibration(
        self, 
        calib_logits: torch.Tensor,
        calib_targets: torch.Tensor
    ):
        """Setup conformal calibration using pre-computed logits."""
        
        if self.prediction_method.needs_calibration():
            self.Qhat = self._conformal_calibration(calib_logits, calib_targets)
        else:
            self.Qhat = None
    
    def _conformal_calibration(
        self, 
        calib_logits: torch.Tensor, 
        calib_targets: torch.Tensor
    ) -> float:
        """Perform conformal calibration to compute the threshold."""
        if self.verbose:
            print("Performing conformal calibration...")
            
        conformity_scores = []
        p_hats = self.predict_prob(calib_logits)

        for i in range(p_hats.shape[0]):
            p_hat = p_hats[i]
            target = calib_targets[i].item()
            
            score = self.prediction_method.compute_conformity_score(p_hat, target)
            conformity_scores.append(score)

        conformity_scores = np.sort(conformity_scores)
        n = len(conformity_scores)
        
        if self.prediction_method.is_conformity_score():
            k = int(np.ceil((n + 1) * self.alpha))
            Qhat = conformity_scores[k - 1]
        else:
            k = int(np.ceil((n + 1) * (1 - self.alpha)))
            Qhat = conformity_scores[k - 1]
        
        if self.verbose:
            score_type = "conformity" if self.prediction_method.is_conformity_score() else "non-conformity"
            quantile = self.alpha if self.prediction_method.is_conformity_score() else (1 - self.alpha)
            print(f"Using {score_type} score with {quantile:.3f} quantile")
            print(f"Calibration threshold (Qhat): {Qhat:.6f}")
            
        return Qhat
    
    def predict(self, test_logits: torch.Tensor) -> List[np.ndarray]:
        """
        Generate prediction sets for test logits.
        """
        p_hats = self.predict_prob(test_logits)
        prediction_sets = []

        for p_hat in p_hats:
            prediction_set = self.prediction_method.compute_prediction_set(p_hat, self.Qhat)
            prediction_sets.append(prediction_set)

        return prediction_sets
    
    def predict_prob(self, test_logits: torch.Tensor) -> np.ndarray:
        """
        Convert logits to calibrated probabilities.
        """
        return softmax(test_logits.numpy() / self.temperature, axis=1)
    
    def get_method_info(self) -> dict:
        """Get information about the current conformal method."""
        return {
            'method_name': self.method_name,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'threshold': self.Qhat if self.Qhat is not None else None,
        }
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available conformal prediction methods."""
        return MethodFactory.get_available_methods()


# ============================================================================
# High-Level Wrapper for Backward Compatibility
# ============================================================================

class ConformalModel(nn.Module):
    """
    High-level wrapper that maintains backward compatibility.
    This class wraps a base model with conformal prediction capabilities.
    
    For new code, consider using ConformalPredictor directly for better flexibility.
    
    Args:
        model: The base neural network model
        calib_loader: DataLoader for calibration data
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        method: Conformal prediction method name
        T: Temperature parameter for Platt scaling (optional)
        calib_logits: Pre-computed calibration logits (optional)
        verbose: Whether to print progress information
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        calib_loader: tdata.DataLoader, 
        alpha: float, 
        method: str = 'pure-singleton',
        T: Optional[torch.Tensor] = None,
        calib_logits: Optional[tdata.TensorDataset] = None,
        verbose: bool = True,
        **kwargs
    ):
        super(ConformalModel, self).__init__()
        self.model = model
        self.verbose = verbose
        
        if T is None or calib_logits is None:
            self.T, calib_logits = platt_scale(self.model, calib_loader, verbose=self.verbose)
        else:
            self.T = T
        
        self.conformal_predictor = ConformalPredictor(
            calib_logits=calib_logits, 
            alpha=alpha,
            method=method,
            temperature=self.T.item(),
            verbose=verbose,
            **kwargs
        )
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Forward pass that returns both logits and prediction sets.
        """
        logits = self.model(*args, **kwargs)
        prediction_sets = self.conformal_predictor.predict(logits.detach().cpu())
        return logits, prediction_sets
    
    def predict_with_sets(
        self, 
        dataloader: tdata.DataLoader
    ) -> Tuple[List[torch.Tensor], List[List[np.ndarray]]]:
        """
        Generate predictions and prediction sets for a dataset.
        """
        self.eval()
        all_logits = []
        all_prediction_sets = []
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader, disable=not self.verbose, desc="Predicting"):
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 1:
                    inputs = batch_data[0]
                else:
                    inputs = batch_data
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                logits, prediction_sets = self.forward(inputs)
                all_logits.append(logits.cpu())
                all_prediction_sets.append(prediction_sets)
        
        return all_logits, all_prediction_sets
    
    def get_method_info(self) -> dict:
        """Get information about the current conformal method."""
        return self.conformal_predictor.get_method_info()
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available conformal prediction methods."""
        return MethodFactory.get_available_methods()
