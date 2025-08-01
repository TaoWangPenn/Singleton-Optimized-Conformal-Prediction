# src/conformalizer.py

import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import torch.utils.data as tdata
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from .utils import get_logits_targets, platt_logits, platt_scale
from .optimization.M_criterion import conjecture_gamma

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
    def is_randomized(self) -> bool:
        """Return True if this method uses randomization."""
        pass
    
    def needs_calibration(self) -> bool:
        """Return True if this method needs conformity score calibration."""
        return True


# ============================================================================
# M-Criterion Methods
# ============================================================================

class MCriterionBase(PredictionSetMethod):
    """Base class for M-criterion methods with shared gamma computation."""
    
    def _get_gamma_m_criterion(self, p_hat: np.ndarray) -> np.ndarray:
        max_p_hat = np.max(p_hat)
        one_minus_alpha = 1 - self.alpha

        if max_p_hat >= one_minus_alpha:
            gamma = np.zeros_like(p_hat)
            gamma[np.argmax(p_hat)] = (1 - self.alpha) / max_p_hat
            return gamma
        else:
            p_hat_sorted_indices = np.argsort(p_hat)[::-1]
            p_hat_sorted = p_hat[p_hat_sorted_indices]
            gamma_conjecture_sorted = conjecture_gamma(p_hat_sorted, self.alpha)
            gamma_conjecture = np.zeros_like(gamma_conjecture_sorted)
            gamma_conjecture[p_hat_sorted_indices] = gamma_conjecture_sorted
            return gamma_conjecture


class MCriterionSimple(MCriterionBase):
    """M-criterion simple conformal prediction method."""
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        gamma = self._get_gamma_m_criterion(p_hat)
        return gamma[target]
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        gamma = self._get_gamma_m_criterion(p_hat)
        return np.where(gamma >= threshold)[0]
    
    def is_randomized(self) -> bool:
        return False


class MCriterionWeighted(MCriterionBase):
    """M-criterion weighted conformal prediction method."""
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        gamma = self._get_gamma_m_criterion(p_hat)
        return p_hat[target] * gamma[target]
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        gamma = self._get_gamma_m_criterion(p_hat)
        return np.where(p_hat * gamma > threshold)[0]
    
    def is_randomized(self) -> bool:
        return False


# ============================================================================
# Length Optimization Methods
# ============================================================================

class LengthOptimizationBase(PredictionSetMethod):
    """Base class for length optimization methods with shared gamma computation."""

    def _get_gamma_lo(self, p_hat: np.ndarray) -> np.ndarray:
        p_hat_sorted_indices = np.argsort(p_hat)[::-1]
        p_hat_sorted = p_hat[p_hat_sorted_indices]
        
        p_hat_cumulative = np.cumsum(p_hat_sorted)
        
        s_search = np.where(p_hat_cumulative >= (1 - self.alpha))[0]
        s = len(p_hat) if len(s_search) == 0 else s_search[0] + 1

        delta = (1 - self.alpha) - (p_hat_cumulative[s - 2] if s > 1 else 0)

        gamma_sorted = np.zeros_like(p_hat_sorted)
        gamma_sorted[:s-1] = 1.0
        if s <= len(p_hat_sorted) and p_hat_sorted[s-1] > 0:
            gamma_sorted[s-1] = delta / p_hat_sorted[s-1]
        
        gamma = np.zeros_like(p_hat)
        gamma[p_hat_sorted_indices] = gamma_sorted
        return gamma


class LengthOptimization(LengthOptimizationBase):
    """Length optimization conformal prediction method."""
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        gamma = self._get_gamma_lo(p_hat)
        return gamma[target]
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        gamma = self._get_gamma_lo(p_hat)
        return np.where(gamma >= threshold)[0]
    
    def is_randomized(self) -> bool:
        return False


class LengthOptimizationWeighted(LengthOptimizationBase):
    """Length-optimization weighted conformal prediction method."""
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        gamma = self._get_gamma_lo(p_hat)
        return p_hat[target] * gamma[target]
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float) -> np.ndarray:
        gamma = self._get_gamma_lo(p_hat)
        return np.where(p_hat * gamma > threshold)[0]
    
    def is_randomized(self) -> bool:
        return False


# ============================================================================
# Randomized Methods
# ============================================================================

class MCriterionRandomized(MCriterionBase):
    """Randomized M-criterion conformal prediction method."""
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        gamma = self._get_gamma_m_criterion(p_hat)
        return gamma[target]
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float = None) -> np.ndarray:
        gamma = self._get_gamma_m_criterion(p_hat)
        samples = np.random.random(len(gamma)) < gamma
        return np.where(samples)[0]
    
    def is_randomized(self) -> bool:
        return True


class LengthOptimizationRandomized(LengthOptimizationBase):
    """Randomized length optimization conformal prediction method."""
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        gamma = self._get_gamma_lo(p_hat)
        return gamma[target]
    
    def compute_prediction_set(self, p_hat: np.ndarray, threshold: float = None) -> np.ndarray:
        gamma = self._get_gamma_lo(p_hat)
        samples = np.random.random(len(gamma)) < gamma
        return np.where(samples)[0]
    
    def is_randomized(self) -> bool:
        return True


# ============================================================================
# Naive Conformal Method
# ============================================================================

class NaiveConformal(PredictionSetMethod):
    """Naive conformal prediction method."""
    
    def compute_conformity_score(self, p_hat: np.ndarray, target: int) -> float:
        raise NotImplementedError("Naive method doesn't use conformity scores")
    
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
    
    def is_randomized(self) -> bool:
        return False
    
    def needs_calibration(self) -> bool:
        return False


# ============================================================================
# Factory Pattern for Method Creation
# ============================================================================

class MethodFactory:
    """Factory class for creating prediction set methods."""
    
    _methods = {
        'm-criterion-simple': MCriterionSimple,
        'm-criterion-weighted': MCriterionWeighted,
        'length-optimization': LengthOptimization,
        'length-optimization-weighted': LengthOptimizationWeighted,
        'm-criterion-randomized': MCriterionRandomized,
        'length-optimization-randomized': LengthOptimizationRandomized,
        'naive': NaiveConformal,
    }
    
    @classmethod
    def create_method(cls, method_name: str, alpha: float) -> PredictionSetMethod:
        """Create a prediction set method instance."""
        if method_name not in cls._methods:
            available = list(cls._methods.keys())
            raise ValueError(f"Unknown method '{method_name}'. Available methods: {available}")
        
        return cls._methods[method_name](alpha)
    
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
        method: str = 'm-criterion-simple',
        temperature: float = 1.0,
        verbose: bool = True
    ):
        self.alpha = alpha
        self.method_name = method
        self.temperature = temperature
        self.verbose = verbose
        
        if not isinstance(calib_logits, tdata.TensorDataset):
            raise ValueError("calib_logits must be a TensorDataset (use get_logits_targets to create it)")
        
        calib_logits_tensor = calib_logits.tensors[0]
        calib_targets_tensor = calib_logits.tensors[1]
        
        self.prediction_method = MethodFactory.create_method(method, alpha)
        self._setup_calibration(calib_logits_tensor, calib_targets_tensor)
    
    def _setup_calibration(
        self, 
        calib_logits: torch.Tensor,
        calib_targets: torch.Tensor
    ):
        """Setup conformal calibration using pre-computed logits."""
        
        if not self.prediction_method.is_randomized() and self.prediction_method.needs_calibration():
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
        k = int(np.ceil((n + 1) * self.alpha))
        Qhat = conformity_scores[k - 1]
        
        if self.verbose:
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
            'is_randomized': self.prediction_method.is_randomized(),
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
        method: str = 'm-criterion-simple',
        T: Optional[torch.Tensor] = None,
        calib_logits: Optional[tdata.TensorDataset] = None,
        verbose: bool = True
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
            verbose=verbose
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

