import argparse
import time
import numpy as np
import torch
import pickle
from pathlib import Path
import sys
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.conformalizer import MethodFactory
from src.utils import platt_logits
from CPL.utils import validate_cpl, split_reps_logits, set_random_seeds, split_for_fullbatch
from CPL.cpl_framework_fullbatch import CPLConformalizerFullBatch

def main():
    parser = argparse.ArgumentParser(description='Evaluate CPL (Full-Batch) with separate theta/c calibration splits')
    parser.add_argument('--logits_file', type=str, default='MedMNIST/logits/tissuemnist_logits.pkl', help='Path to pre-computed logits file (.pkl)')
    parser.add_argument('--reps_file', type=str, default='/path/to/your/CPL_HiddenRep/medmnist/tissuemnist_hidden_reps.pkl', help='Path to pre-computed hidden representations file (.pkl)')

    parser.add_argument('--num_tune', type=int, default=10000, help='Number of samples for temperature scaling')
    parser.add_argument('--num_calib', type=int, default=15000, help='Number of calibration points for CPL')
    parser.add_argument('--num_val', type=int, default=15000, help='Number of validation points')
    parser.add_argument('--theta_split_ratio', type=float, default=0.5, help='Fraction of calibration used for theta updates (rest for c)')

    parser.add_argument('--alpha', type=float, default=0.05, help='Error rate')
    parser.add_argument('--score_method', type=str, default='las', help='Method to compute non-conformity scores')
    parser.add_argument('--temperature', type=float, default=None, help='Temperature for scaling (default: auto-tune)')

    parser.add_argument('--cpl_sigma', type=float, default=0.1, help='Initial smoothing parameter for CPL')
    parser.add_argument('--cpl_sigma_min', type=float, default=0.01, help='Minimum sigma value for annealing')
    parser.add_argument('--cpl_sigma_annealing', type=float, default=0.95, help='Sigma annealing factor per epoch')
    parser.add_argument('--cpl_lr_theta', type=float, default=0.001, help='Learning rate for CPL h_model optimizer')
    parser.add_argument('--cpl_lr_c', type=float, default=0.1, help='Learning rate for CPL c optimizer')
    parser.add_argument('--cpl_n_epochs', type=int, default=100, help='Number of training epochs for CPL')
    parser.add_argument('--cpl_inner_steps', type=int, default=20, help='Inner loop steps for CPL')
    parser.add_argument('--cpl_outer_steps', type=int, default=1, help='Outer loop steps for CPL')
    parser.add_argument('--cpl_use_lr_scheduler', action='store_true', help='Use LR schedulers for full-batch CPL')
    parser.add_argument('--cpl_lr_scheduler_step_size', type=int, default=5, help='LR scheduler step size')
    parser.add_argument('--cpl_lr_scheduler_gamma', type=float, default=0.5, help='LR scheduler gamma')

    parser.add_argument('--repetitions', type=int, default=5, help='Number of repetitions for averaging results')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    args = parser.parse_args()

    set_random_seeds(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading logits from: {args.logits_file}")
    with open(args.logits_file, 'rb') as f:
        full_logits_dataset = pickle.load(f)

    print(f"Loading representations from: {args.reps_file}")
    with open(args.reps_file, 'rb') as f:
        full_reps_dataset = pickle.load(f)

    print(f"Loaded {full_logits_dataset.tensors[0].shape[0]} samples.")

    method_results = []
    method_times = []

    for run in range(args.repetitions):
        print(f"\n--- Run {run + 1}/{args.repetitions} ---")
        start_time = time.time()

        current_seed = args.seed + run * 100
        set_random_seeds(current_seed)

        tune_data, calib_data, val_data = split_reps_logits(
            full_reps_dataset,
            full_logits_dataset,
            num_tune=args.num_tune,
            num_calib=args.num_calib,
            num_val=args.num_val,
        )

        calib_reps, calib_logits, calib_labels = calib_data
        val_reps, val_logits, val_labels = val_data

        if args.temperature is not None:
            temperature = args.temperature
            print(f"Using fixed temperature: {temperature}")
        else:
            print("Tuning temperature on tuning set...")
            calib_loader = DataLoader(TensorDataset(calib_logits, calib_labels), batch_size=512)
            T_param = platt_logits(calib_loader)
            temperature = T_param.item()
            print(f"Auto-tuned temperature: {temperature:.4f}")

        print(f"Computing non-conformity scores using '{args.score_method}' method...")
        score_computer = MethodFactory.create_method(args.score_method, args.alpha)

        calib_probs = torch.softmax(calib_logits / temperature, dim=1).numpy()
        val_probs = torch.softmax(val_logits / temperature, dim=1).numpy()

        if args.score_method == 'las':
            calib_labels_np = calib_labels.numpy()
            calib_scores_true_raw = calib_probs[np.arange(calib_probs.shape[0]), calib_labels_np]
            calib_scores_all_raw = calib_probs
            val_scores_all_raw = val_probs
        else:
            calib_scores_true_raw = np.array([
                score_computer.compute_conformity_score(p, t) for p, t in zip(calib_probs, calib_labels.numpy())
            ])
            all_labels = torch.arange(calib_logits.shape[1])
            calib_scores_all_raw = np.array([[score_computer.compute_conformity_score(p, y) for y in all_labels] for p in calib_probs])
            val_scores_all_raw = np.array([[score_computer.compute_conformity_score(p, y) for y in all_labels] for p in val_probs])

        if score_computer.is_conformity_score():
            print("Transforming conformity scores to non-conformity scores.")
            calib_scores_true = 1 - calib_scores_true_raw
            calib_scores_all = 1 - calib_scores_all_raw
            val_scores_all = 1 - val_scores_all_raw
        else:
            calib_scores_true = calib_scores_true_raw
            calib_scores_all = calib_scores_all_raw
            val_scores_all = val_scores_all_raw

        (theta_reps_t, theta_scores_true_t, theta_scores_all_t,
         c_reps_t, c_scores_true_t, c_scores_all_t) = split_for_fullbatch(
            torch.from_numpy(calib_reps).float(),
            torch.from_numpy(calib_scores_true).float(),
            torch.from_numpy(calib_scores_all).float(),
            split_ratio=args.theta_split_ratio,
        )

        cpl = CPLConformalizerFullBatch(
            alpha=args.alpha,
            sigma=args.cpl_sigma,
            lr_theta=args.cpl_lr_theta,
            lr_c=args.cpl_lr_c,
            n_epochs=args.cpl_n_epochs,
            inner_steps=args.cpl_inner_steps,
            outer_steps=args.cpl_outer_steps,
            clip_value=1.0,
            use_lr_scheduler=args.cpl_use_lr_scheduler,
            lr_scheduler_step_size=args.cpl_lr_scheduler_step_size,
            lr_scheduler_gamma=args.cpl_lr_scheduler_gamma,
            sigma_annealing_factor=args.cpl_sigma_annealing,
            sigma_min=args.cpl_sigma_min,
            device=device,
        )

        cpl.fit(
            theta_reps_t, theta_scores_true_t, theta_scores_all_t,
            c_reps_t, c_scores_true_t, c_scores_all_t,
        )

        prediction_sets = cpl.predict(
            torch.from_numpy(val_reps).float(),
            torch.from_numpy(val_scores_all).float(),
        )

        metrics = validate_cpl(prediction_sets, val_logits, val_labels)
        elapsed_time = time.time() - start_time
        
        method_results.append(metrics)
        method_times.append(elapsed_time)

        print(
            f"Run {run + 1} completed in {elapsed_time:.3f}s - "
            f"Top-1: {metrics['Top-1 Accuracy']:.4f}, Coverage: {metrics['Coverage']:.4f}, "
            f"Avg Size: {metrics['Average Set Size']:.4f}, P(size>1): {metrics['Prob(Size > 1)']:.4f}, "
            f"P(Empty): {metrics['Prob(Empty)']:.4f}"
        )

    all_metrics = ['Top-1 Accuracy', 'Top-5 Accuracy', 'Coverage', 'Average Set Size', 'Prob(Size > 1)', 'Prob(Empty)']
    avg_results = {}
    std_results = {}
    for metric in all_metrics:
        values = [r[metric] for r in method_results]
        avg_results[metric] = np.mean(values)
        std_results[metric] = np.std(values)

    avg_time = np.mean(method_times)
    std_time = np.std(method_times)

    print("\n" + "="*100)
    print(f"FINAL CPL FULL-BATCH RESULTS ({args.repetitions} repetitions)")
    print("="*100)
    print(f"{'Score Method:':<20} {args.score_method}")
    print("-" * 100)
    print(f"{'Metric':<20} {'Mean':<20} {'Std Dev':<20}")
    print("-" * 100)
    for metric in all_metrics:
        print(f"{metric:<20} {avg_results[metric]:<20.4f} {std_results[metric]:<20.4f}")
    print(f"{'Time per run (s)':<20} {avg_time:<20.3f} {std_time:<20.3f}")
    print("="*100)


if __name__ == '__main__':
    main()
