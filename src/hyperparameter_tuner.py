# src/hyperparameter_tuner.py

import numpy as np
import torch
import torch.utils.data as tdata
from scipy.special import softmax
from typing import Dict, List, Optional
from .conformalizer import ConformalPredictor, MethodFactory
from .utils import validate
from kneed import KneeLocator


class HyperparameterTuner:
    """
    Hyperparameter tuner for conformal prediction methods.
    """

    def __init__(self, alpha: float, temperature: float = 1.0, verbose: bool = True, randomized: bool = False, use_numba: bool = False):
        self.alpha = alpha
        self.temperature = temperature
        self.verbose = verbose
        self.randomized = randomized
        self.use_numba = use_numba
    
    def tune_raps_hyperparameters(self, tune_dataset: tdata.TensorDataset) -> Dict:
        """
        Tune RAPS hyperparameters using the GET-KSTAR procedure and λ optimization.
        """
        if self.verbose:
            print("Tuning RAPS hyperparameters...")
        
        tune_logits_tensor = tune_dataset.tensors[0]
        tune_targets_tensor = tune_dataset.tensors[1]
        tune_logits = tune_logits_tensor.detach().cpu().numpy()
        tune_probs = softmax(tune_logits / self.temperature, axis=1)
        
        n = len(tune_probs)
        
        L_values = []
        for i in range(n):
            p_hat = tune_probs[i]
            y_i = tune_targets_tensor[i].item()
            I = np.argsort(p_hat)[::-1]
            L_i = np.where(I == y_i)[0][0] + 1
            L_values.append(L_i)
        
        L_values_arr = np.asarray(L_values)
        L_sorted_desc = np.sort(L_values_arr)[::-1]
        
        n = len(L_values_arr)
        k = int(np.ceil((1.0 - self.alpha) * (n + 1)))
        k = min(max(k, 1), n) 
        k_star = int(L_sorted_desc[k - 1])


        if self.verbose:
            print(f"Computed k* = {k_star}")
        
        lambda_candidates = [0.001, 0.01, 0.1, 0.2, 0.5]
        
        best_lambda = 0.01
        best_avg_size = float('inf')
        tune_loader = tdata.DataLoader(tune_dataset, batch_size=64, shuffle=False)

        for penalty_lambda in lambda_candidates:
            try:
                conformal_predictor = ConformalPredictor(
                    tune_dataset,
                    alpha=self.alpha,
                    method='raps', 
                    k_reg=int(k_star), 
                    penalty_lambda=penalty_lambda,
                    randomized=self.randomized, 
                    temperature=self.temperature,
                    verbose=False
                )
                
                _, _, _, avg_size, _, _ = validate(
                    tune_loader, 
                    conformal_predictor, 
                    val_logits=tune_dataset, 
                    print_bool=False
                )
                
                if self.verbose:
                    print(f"λ={penalty_lambda}: avg set size = {avg_size:.3f}")
                
                if avg_size < best_avg_size:
                    best_avg_size = avg_size
                    best_lambda = penalty_lambda
                    
            except Exception as e:
                if self.verbose:
                    print(f"Failed for λ={penalty_lambda}: {e}")
                continue
        
        best_params = {'k_reg': int(k_star), 'penalty_lambda': best_lambda}
        
        if self.verbose:
            print(f"Best RAPS parameters: k_reg={k_star}, penalty_lambda={best_lambda} (avg size: {best_avg_size:.3f})")
        
        return best_params
    
    def tune_socop_hyperparameters(self, tune_dataset: tdata.TensorDataset, method: str) -> Dict:
        """
        Tune hyperparameters for SOCOP method using knee point detection.
        """
        if self.verbose:
            print(f"Tuning {method} hyperparameters...")

        lambda_candidates = np.linspace(0.05, 1.0, 15) # np.linspace(0.05, 1.0, 15) for 0.95, np.linspace(0.08, 1.0, 15) for 0.99
        
        avg_sizes = []
        p_gt1s = []
        
        val_loader = tdata.DataLoader(tune_dataset, batch_size=64, shuffle=False)

        for penalty_lambda in lambda_candidates:
            try:
                predictor_kwargs = {
                    'alpha': self.alpha,
                    'method': method,
                    'penalty_lambda': penalty_lambda,
                    'verbose': False
                }
                if method == 'socop':
                    predictor_kwargs['use_numba'] = self.use_numba

                conformal_predictor = ConformalPredictor(
                    tune_dataset,
                    **predictor_kwargs
                )
                
                _, _, _, avg_size, p_gt1, _ = validate(
                    val_loader, 
                    conformal_predictor, 
                    val_logits=tune_dataset, 
                    print_bool=False
                )
                
                avg_sizes.append(avg_size)
                p_gt1s.append(p_gt1)

            except Exception as e:
                if self.verbose:
                    print(f"Failed for λ={penalty_lambda}: {e}")
                avg_sizes.append(np.inf)
                p_gt1s.append(1.0)

        # Find the knee point
        try:
            sorted_indices = np.argsort(p_gt1s)
            sorted_p_gt1s = np.array(p_gt1s)[sorted_indices]
            sorted_avg_sizes = np.array(avg_sizes)[sorted_indices]
            
            kn = KneeLocator(
                sorted_p_gt1s, 
                sorted_avg_sizes, 
                curve='convex', 
                direction='decreasing',
                interp_method='polynomial',
                polynomial_degree=2
            )
            
            if kn.knee is None:
                if self.verbose:
                    print("Could not find a knee point. Defaulting to lambda with smallest size.")
                best_lambda_idx = np.argmin(avg_sizes)
                best_lambda = lambda_candidates[best_lambda_idx]
            else:
                knee_p_gt1 = kn.knee
                best_lambda_idx = np.argmin(np.abs(np.array(p_gt1s) - knee_p_gt1))
                best_lambda = lambda_candidates[best_lambda_idx]
                if self.verbose:
                    print(f"Knee point found at p_gt1={knee_p_gt1:.4f}, corresponding to λ={best_lambda:.4f}")

        except Exception as e:
            if self.verbose:
                print(f"Knee point detection failed: {e}. Defaulting to lambda with smallest size.")
            best_lambda_idx = np.argmin(avg_sizes)
            best_lambda = lambda_candidates[best_lambda_idx]

        best_params = {'penalty_lambda': best_lambda}
        
        if self.verbose:
            print(f"Best {method} parameters: penalty_lambda={best_lambda:.4f}")
            
        return best_params
    
    def get_tuned_parameters(self, method: str, tune_dataset: tdata.TensorDataset, args) -> Dict:
        """
        Get tuned parameters for a method, using provided parameters if available, otherwise auto-tune.
        """
        if method == 'raps':
            return self._handle_raps_parameters(tune_dataset, args)
        elif method in ['socop']:
            return self._handle_socop_parameters(tune_dataset, method, args)
        else:
            return {}
    
    def _handle_raps_parameters(self, tune_dataset: tdata.TensorDataset, args) -> Dict:
        """Handle RAPS parameter tuning logic."""
        k_reg_provided = hasattr(args, 'k_reg') and args.k_reg is not None
        lambda_provided = hasattr(args, 'penalty_lambda') and args.penalty_lambda is not None
        
        if k_reg_provided and lambda_provided:
            return {'k_reg': args.k_reg, 'penalty_lambda': args.penalty_lambda}
        else:
            tuned_params = self.tune_raps_hyperparameters(tune_dataset)
            if k_reg_provided:
                tuned_params['k_reg'] = args.k_reg
            if lambda_provided:
                tuned_params['penalty_lambda'] = args.penalty_lambda
            return tuned_params
    
    def _handle_socop_parameters(self, tune_dataset: tdata.TensorDataset, method: str, args) -> Dict:
        """Handle SOCOP parameter tuning logic."""
        lambda_provided = hasattr(args, 'penalty_lambda') and args.penalty_lambda is not None
        
        final_params = {}
        if lambda_provided:
            final_params['penalty_lambda'] = args.penalty_lambda
        else:
            tuned_params = self.tune_socop_hyperparameters(tune_dataset, method)
            final_params.update(tuned_params)
            
        return final_params
    
    @classmethod
    def needs_tuning(cls, method: str) -> bool:
        """Check if a method supports hyperparameter tuning by checking if it has a handler."""
        return method in ['raps', 'socop']
