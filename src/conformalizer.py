import torch
import torch.nn as nn
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import torch.utils.data as tdata

from .utils import get_logits_targets, platt_logits
from .optimization.M_criterion import (
    OptimizationProblem, 
    SLSQPSolver, 
    find_best_solution_with_multistart, 
    conjecture_gamma
)

class ConformalModel(nn.Module):
    def __init__(self, model, calib_loader, alpha, batch_size=32, threshold=0.001, num_starts_slsqp=5):
        super(ConformalModel, self).__init__()
        self.model = model
        self.alpha = alpha
        self.threshold = threshold
        self.num_starts_slsqp = num_starts_slsqp 
        self.T, calib_logits = self.platt(calib_loader)
        # self.num_classes = len(calib_loader.dataset.dataset.classes) if hasattr(calib_loader.dataset, 'dataset') else calib_loader.dataset[0][0].shape[0]

        calib_loader = tdata.DataLoader(calib_logits, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.Qhat = self.conformal_calibration(calib_loader)

    def platt(self, calib_loader):
        print("Begin Platt scaling.")
        logits_dataset = get_logits_targets(self.model, calib_loader)
        logits_loader = torch.utils.data.DataLoader(logits_dataset, batch_size=calib_loader.batch_size, shuffle=False, pin_memory=True)

        T = platt_logits(logits_loader)

        print(f"Optimal T={T.item()}")
        return T, logits_dataset

    def get_gamma(self, p_hat: np.ndarray) -> np.ndarray:
        max_p_hat = np.max(p_hat)
        one_minus_alpha = 1 - self.alpha

        if max_p_hat >= one_minus_alpha:
            gamma = np.zeros_like(p_hat)
            gamma[np.argmax(p_hat)] = 1.0
            return gamma

        p_hat_sorted_indices = np.argsort(p_hat)[::-1]
        p_hat_sorted = p_hat[p_hat_sorted_indices]
        gamma_conjecture_sorted = conjecture_gamma(p_hat_sorted, self.alpha)
        gamma_conjecture = np.zeros_like(gamma_conjecture_sorted)
        gamma_conjecture[p_hat_sorted_indices] = gamma_conjecture_sorted

        if (one_minus_alpha / 2) < max_p_hat < one_minus_alpha:
            return gamma_conjecture

        else: 
            problem = OptimizationProblem(p_hat, self.alpha, self.threshold)

            slsqp_solver = SLSQPSolver(problem, tol=1e-9)
            best_result_slsqp = find_best_solution_with_multistart(problem, slsqp_solver, num_starts=self.num_starts_slsqp)
            gamma_slsqp = best_result_slsqp.solution

            if gamma_slsqp is None:
                return gamma_conjecture

            log_g_slsqp = best_result_slsqp.value
            log_g_conjecture = problem.objective(gamma_conjecture) 

            if log_g_conjecture >= log_g_slsqp:
                return gamma_conjecture
            else:
                return gamma_slsqp

    def conformal_calibration(self, calib_loader):
        print("Conformal calibration")
        conformity_scores = []
        with torch.no_grad():
            for logits, targets in tqdm(calib_loader):
                p_hats = softmax(logits.numpy() / self.T.item(), axis=1)
                
                for i in range(p_hats.shape[0]):
                    p_hat = p_hats[i]
                    gamma = self.get_gamma(p_hat)
                    
                    target = targets[i]
                    conformity_score = gamma[target]
                    conformity_scores.append(conformity_score)

        conformity_scores = np.array(conformity_scores)
        q_level = np.ceil((1 - self.alpha) * (len(conformity_scores) + 1)) / (len(conformity_scores))
        Qhat = np.quantile(conformity_scores, q_level, interpolation='higher')
        return Qhat

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs)
        
        with torch.no_grad():
            p_hats = softmax(logits.detach().cpu().numpy() / self.T.item(), axis=1)
            prediction_sets = []
            for p_hat in p_hats:
                gamma = self.get_gamma(p_hat)
                prediction_set = np.where(gamma >= self.Qhat)[0]
                prediction_sets.append(prediction_set)

        return logits, prediction_sets
