import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class CPLConformalizerFullBatch:
    """
    Full-batch CPL: optimize h(x;θ) and c using two disjoint calibration splits.
    - Split A (theta-set): used to update θ (maximize objective)
    - Split B (c-set): used to update c (minimize objective)
    No minibatches are used; each step processes the full corresponding split.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        sigma: float = 0.1,
        lr_theta: float = 1e-3,
        lr_c: float = 1e-2,
        n_epochs: int = 30,
        inner_steps: int = 10,
        outer_steps: int = 1,
        clip_value: float = 1.0,
        use_lr_scheduler: bool = True,
        lr_scheduler_step_size: int = 5,
        lr_scheduler_gamma: float = 0.5,
        sigma_annealing_factor: float = 0.95,
        sigma_min: float = 1e-3,
        device=None,
    ):
        self.alpha = alpha
        self.initial_sigma = sigma
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.lr_theta = lr_theta
        self.lr_c = lr_c
        self.n_epochs = n_epochs
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps
        self.clip_value = clip_value
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_step_size = lr_scheduler_step_size
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.sigma_annealing_factor = sigma_annealing_factor
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.c = nn.Parameter(torch.tensor([1.0], device=self.device))
        self.h_model = None
        self.sqrt2 = torch.sqrt(torch.tensor(2.0, device=self.device))

    def _smooth_indicator(self, a, b, tau=None):
        if tau is None:
            tau = self.sigma * 0.6266571
        return torch.sigmoid((b - a) / tau)

    def _g_alpha_n(self, c, h_vals, scores_true, scores_all):
        smooth_coverage = self._smooth_indicator(scores_true, h_vals)
        coverage_term = c * (smooth_coverage - (1 - self.alpha))

        h_vals_expanded = h_vals.unsqueeze(1)
        smooth_length = self._smooth_indicator(scores_all, h_vals_expanded).mean(dim=1)

        objective = (coverage_term - smooth_length).mean()

        with torch.no_grad():
            real_size = (scores_all <= h_vals_expanded).float().sum(dim=1)

        diagnostics = {
            "smooth_gap": (smooth_coverage.mean().item() - (1 - self.alpha)),
            "smooth_size": smooth_length.mean().item(),
            "real_size": real_size.mean().item(),
        }
        return objective, diagnostics

    def fit(
        self,
        theta_reps: torch.Tensor,
        theta_scores_true: torch.Tensor,
        theta_scores_all: torch.Tensor,
        c_reps: torch.Tensor,
        c_scores_true: torch.Tensor,
        c_scores_all: torch.Tensor,
    ):
        """
        Full-batch alternating optimization using two fixed calibration splits.
        """
        self.sigma = self.initial_sigma

        theta_reps = theta_reps.to(self.device)
        theta_scores_true = theta_scores_true.to(self.device)
        theta_scores_all = theta_scores_all.to(self.device)

        c_reps = c_reps.to(self.device)
        c_scores_true = c_scores_true.to(self.device)
        c_scores_all = c_scores_all.to(self.device)

        d = theta_reps.shape[1]
        optimizer_c = optim.SGD([self.c], lr=self.lr_c)
        scheduler_c = optim.lr_scheduler.StepLR(
            optimizer_c, step_size=self.lr_scheduler_step_size, gamma=self.lr_scheduler_gamma
        )

        print("Fitting CPL (full-batch) model...")
        for epoch in range(self.n_epochs):
            self.h_model = nn.Sequential(
                nn.Linear(d, 1),
                nn.Sigmoid(),
            ).to(self.device)

            with torch.no_grad():
                h0 = torch.quantile(theta_scores_true, q=1.0 - self.alpha)
                h0 = h0.clamp_(1e-6, 1.0 - 1e-6)
                b0 = torch.log(h0 / (1.0 - h0))
                self.h_model[0].bias.fill_(b0)
                self.h_model[0].weight.fill_(0.0)
                if epoch == 0:
                    print(f"Initial h0 quantile: {h0.item():.4f}, Initial bias b0: {b0.item():.4f}")

            optimizer_theta = optim.Adam(self.h_model.parameters(), lr=self.lr_theta)
            scheduler_theta = optim.lr_scheduler.StepLR(
                optimizer_theta, step_size=self.lr_scheduler_step_size, gamma=self.lr_scheduler_gamma
            )
            epoch_diag_theta = {"smooth_gap": [], "smooth_size": [], "real_size": []}
            epoch_diag_c = {"smooth_gap": [], "smooth_size": [], "real_size": []}

            for _ in range(self.inner_steps):
                optimizer_theta.zero_grad()
                h_vals_theta = self.h_model(theta_reps).squeeze(-1)
                loss_theta, diag_theta = self._g_alpha_n(self.c.detach(), h_vals_theta, theta_scores_true, theta_scores_all)
                (-loss_theta).backward()
                nn.utils.clip_grad_norm_(list(self.h_model.parameters()), self.clip_value)
                optimizer_theta.step()
                epoch_diag_theta["smooth_gap"].append(diag_theta["smooth_gap"])    
                epoch_diag_theta["smooth_size"].append(diag_theta["smooth_size"])  
                epoch_diag_theta["real_size"].append(diag_theta["real_size"])      

            for _ in range(self.outer_steps):
                optimizer_c.zero_grad()
                with torch.no_grad():
                    h_vals_c = self.h_model(c_reps).squeeze(-1)
                loss_c, diag_c = self._g_alpha_n(self.c, h_vals_c, c_scores_true, c_scores_all)
                loss_c.backward()
                nn.utils.clip_grad_norm_([self.c], self.clip_value)
                optimizer_c.step()
                self.c.data.clamp_(min=0.0)
                epoch_diag_c["smooth_gap"].append(diag_c["smooth_gap"])      
                epoch_diag_c["smooth_size"].append(diag_c["smooth_size"])    
                epoch_diag_c["real_size"].append(diag_c["real_size"])        

            if self.use_lr_scheduler:
                scheduler_c.step()
                scheduler_theta.step()

            self.sigma = max(self.sigma * self.sigma_annealing_factor, self.sigma_min)

            avg_gap_c = float(np.mean(epoch_diag_c["smooth_gap"])) if epoch_diag_c["smooth_gap"] else float("nan")
            avg_size_c = float(np.mean(epoch_diag_c["smooth_size"])) if epoch_diag_c["smooth_size"] else float("nan")
            avg_real_c = float(np.mean(epoch_diag_c["real_size"])) if epoch_diag_c["real_size"] else float("nan")
            print(
                f"Epoch {epoch+1} complete. Smooth Gap (c-split): {avg_gap_c:.4f}, "
                f"Smooth Size: {avg_size_c:.3f}, c: {self.c.item():.4f}, Real Size: {avg_real_c:.3f}, Sigma: {self.sigma:.4f}"
            )

        print("CPL (full-batch) fitting complete.")
        self.h_model.eval()

    def predict(self, test_reps: torch.Tensor, test_scores_all: torch.Tensor):
        if self.h_model is None:
            raise RuntimeError("The model must be fitted before prediction.")

        test_reps = test_reps.to(self.device)
        test_scores_all = test_scores_all.to(self.device)

        prediction_sets = []
        with torch.no_grad():
            h_vals = self.h_model(test_reps).squeeze(-1)
            for i in range(test_reps.shape[0]):
                h_i = h_vals[i]
                scores_i = test_scores_all[i]
                pred_set = torch.where(scores_i <= h_i)[0].cpu().numpy()
                prediction_sets.append(pred_set)
        return prediction_sets
