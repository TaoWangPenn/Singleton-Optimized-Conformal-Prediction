import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet152_Weights
import numpy as np
import random
import torch.backends.cudnn as cudnn
import time
from pathlib import Path

# Correctly import from your project structure
from src.conformalizer import ConformalModel, ConformalPredictor
from src.utils import validate, get_logits_targets, platt_logits, set_random_seeds
from src.hyperparameter_tuner import HyperparameterTuner


def main():
    """
    Main function to run the conformal prediction example.
    """
    parser = argparse.ArgumentParser(description='Conformal Prediction with a Model and Auto-tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--num_tune', type=int, default=10000, metavar='NTUNE',
                        help='number of samples for hyperparameter tuning (default: 10000)')
    parser.add_argument('--num_calib', type=int, default=20000, metavar='NCALIB', 
                        help='number of calibration points (default: 20000)')
    parser.add_argument('--num_val', type=int, default=20000, metavar='NVAL', 
                        help='number of validation points (default: 20000)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BSZ', 
                        help='batch size for validation (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='NW', 
                        help='number of workers for data loading (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', 
                        help='random seed (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.05, metavar='A', 
                        help='error rate (default: 0.05)')
    parser.add_argument('--temperature', type=float, default=None, metavar='T',
                        help='temperature for scaling (default: auto-tune on calibration set)')
    parser.add_argument('--repetitions', type=int, default=5, metavar='R',
                        help='number of repetitions for averaging results (default: 5)')
    parser.add_argument('--bootstrap_samples', type=int, default=1000, metavar='B',
                        help='number of bootstrap samples for confidence intervals (default: 1000)')
    parser.add_argument('--penalty_lambda', type=float, default=None,
                        help='penalty parameter λ (default: auto-tune)')
    parser.add_argument('--tolerance', type=float, default=1e-8,
                        help='tolerance ε for combined-optimization binary search (default: 1e-8)')
    parser.add_argument('--use_numba', action='store_true',
                        help='Use Numba for the convex hull method')
    parser.add_argument('--k_reg', type=int, default=None,
                        help='k_reg parameter for RAPS method (default: auto-tune)')
    parser.add_argument('--randomized', action='store_true',
                        help='Use randomized version of RAPS')
    parser.add_argument('--methods', nargs='+', default=['combined-optimization-convex-hull'],
                        help='List of conformal methods to run (default: ["combined-optimization-convex-hull"])')

    args = parser.parse_args()

    # --- Initial Reproducibility Setup ---
    set_random_seeds(args.seed)

    # --- Device Configuration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    # Modern transform for ImageNet validation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the full dataset
    full_dataset = torchvision.datasets.ImageFolder(args.data, transform=transform)
    
    # Dynamically get dataset size
    dataset_size = len(full_dataset)
    if args.num_tune + args.num_calib + args.num_val > dataset_size:
        raise ValueError(f"Requested sizes for tune, calib, and val sets exceed dataset size of {dataset_size}.")
    
    # --- Model Loading ---
    print("Loading ResNet-152 model...")
    weights = ResNet152_Weights.IMAGENET1K_V2 # Using V2 for better accuracy
    model = torchvision.models.resnet152(weights=weights).to(device)
    model.eval()

    # --- Conformal Prediction Methods ---
    print(f"\nAvailable conformal methods: {ConformalPredictor.get_available_methods()}")
    methods = args.methods
    print(f"Selected methods to run: {methods}")
    print(f"Running {args.repetitions} repetitions for each method with different data splits...")
    
    if args.temperature is not None:
        print(f"Using fixed temperature: {args.temperature}")
    else:
        print("Auto-tuning temperature on calibration set for each data split...")
    
    all_results_cm = {}
    all_results_cp = {}
    all_timing_results_cm = {}
    all_timing_results_cp = {}
    
    for method in methods:
        all_results_cm[method] = []
        all_results_cp[method] = []
        all_timing_results_cm[method] = []
        all_timing_results_cp[method] = []

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Running method: {method} ({args.repetitions} repetitions)")
        print(f"{'='*80}")
        
        method_results_cm = []
        method_results_cp = []
        method_times_cm = []
        method_times_cp = []
        
        for run in range(args.repetitions):
            print(f"\n--- Run {run + 1}/{args.repetitions} for {method} ---")
            
            start_time_total = time.time()
            
            current_seed = args.seed + run * 100
            set_random_seeds(current_seed)
            
            num_unused = dataset_size - args.num_tune - args.num_calib - args.num_val
            tune_dataset, calib_dataset, val_dataset, _ = torch.utils.data.random_split(
                full_dataset, [args.num_tune, args.num_calib, args.num_val, num_unused]
            )

            tune_loader = torch.utils.data.DataLoader(tune_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

            # --- Pre-compute logits ---
            print("Pre-computing logits for tune, calib, and val sets...")
            tune_logits = get_logits_targets(model, tune_loader)
            calib_logits = get_logits_targets(model, calib_loader)
            val_logits = get_logits_targets(model, val_loader)

            # --- Temperature Scaling ---
            if args.temperature is not None:
                temperature = args.temperature
            else:
                T_param = platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01, logits_dataset=calib_logits)
                temperature = T_param.item()

            # --- Hyperparameter Tuning ---
            if HyperparameterTuner.needs_tuning(method):
                tuner = HyperparameterTuner(args.alpha, temperature, verbose=True, randomized=args.randomized, use_numba=args.use_numba, tolerance=args.tolerance)
                tuned_params = tuner.get_tuned_parameters(method, tune_logits, args)
            else:
                tuned_params = {}
            # =================================================================
            # APPROACH 1: ConformalModel (High-level wrapper)
            # =================================================================
            print(f"\n--- APPROACH 1: ConformalModel (High-level wrapper) ---")
            
            start_time_cm = time.time()
            
            # --- Conformalization ---
            conformal_model = ConformalModel(
                model=model, 
                calib_loader=calib_loader, 
                alpha=args.alpha, 
                method=method, 
                T=temperature, 
                calib_logits=calib_logits,
                verbose=False
            )

            # --- Validation ---
            top1_cm, top5_cm, coverage_cm, size_cm, prob_gt1_cm, prob_empty_cm = validate(val_loader, conformal_model, print_bool=False)
            
            conformal_model_time = time.time() - start_time_cm
            
            run_results_cm = {
                'Top-1 Accuracy': top1_cm,
                'Top-5 Accuracy': top5_cm,
                'Coverage': coverage_cm,
                'Average Set Size': size_cm,
                'Prob(Size > 1)': prob_gt1_cm,
                'Prob(Empty)': prob_empty_cm
            }
            method_results_cm.append(run_results_cm)
            method_times_cm.append(conformal_model_time)
            
            # =================================================================
            # APPROACH 2: ConformalPredictor (Low-level, logits-based)
            # =================================================================
            print(f"\n--- APPROACH 2: ConformalPredictor (Low-level, logits-based) ---")
            
            start_time_cp = time.time()
            
            # --- Create ConformalPredictor ---
            predictor_kwargs = {
                'calib_logits': calib_logits,
                'alpha': args.alpha,
                'method': method,
                'temperature': temperature,
                'verbose': False,
            }
            
            if method == 'raps':
                predictor_kwargs.update(tuned_params)
                predictor_kwargs['randomized'] = args.randomized
            elif method == 'combined-optimization-binary':
                predictor_kwargs.update(tuned_params)
                predictor_kwargs['tolerance'] = args.tolerance
            elif method == 'combined-optimization-convex-hull':
                predictor_kwargs.update(tuned_params)
                predictor_kwargs['use_numba'] = args.use_numba
            
            conformal_predictor = ConformalPredictor(**predictor_kwargs)
            
            # --- Validation ---
            top1_cp, top5_cp, coverage_cp, avg_size_cp, prob_gt1_cp, prob_empty_cp = validate(
                val_loader=val_loader,
                model_or_predictor=conformal_predictor,
                val_logits=val_logits,
                print_bool=False
            )
            
            conformal_predictor_time = time.time() - start_time_cp
            
            run_results_cp = {
                'Top-1 Accuracy': top1_cp,
                'Top-5 Accuracy': top5_cp,
                'Coverage': coverage_cp,
                'Average Set Size': avg_size_cp,
                'Prob(Size > 1)': prob_gt1_cp,
                'Prob(Empty)': prob_empty_cp
            }
            
            method_results_cp.append(run_results_cp)
            method_times_cp.append(conformal_predictor_time)
            
            if tuned_params:
                print(f"Run {run + 1} completed in {time.time() - start_time_total:.3f}s - Coverage: {coverage_cp:.4f}, Avg Size: {avg_size_cp:.4f}, Params: {tuned_params}")
            else:
                print(f"Run {run + 1} completed in {time.time() - start_time_total:.3f}s - Coverage: {coverage_cp:.4f}, Avg Size: {avg_size_cp:.4f}")

        all_results_cm[method] = method_results_cm
        all_results_cp[method] = method_results_cp
        all_timing_results_cm[method] = method_times_cm
        all_timing_results_cp[method] = method_times_cp
        
    # --- Print Final Comparison ---
    print("\n" + "="*100)
    print(f"FINAL COMPARISON OF ALL METHODS (Average across {args.repetitions} different data splits)")
    print("="*100)

    # --- ConformalModel ---
    print("\n--- ConformalModel Results ---")
    results_cm = {}
    results_std_cm = {}
    for method in methods:
        avg_results = {}
        std_results = {}
        for metric in ['Top-1 Accuracy', 'Top-5 Accuracy', 'Coverage', 'Average Set Size', 'Prob(Size > 1)', 'Prob(Empty)']:
            values = [run[metric] for run in all_results_cm[method]]
            avg_results[metric] = np.mean(values)
            std_results[metric] = np.std(values)
        results_cm[method] = avg_results
        results_std_cm[method] = std_results

    print(f"\n{'Method':<30} {'Top-1':<15} {'Top-5':<15} {'Coverage':<15} {'Avg Size':<15} {'P(Size>1)':<15} {'P(Empty)':<15}")
    print("-" * 115)
    for method in methods:
        res = results_cm[method]
        res_std = results_std_cm[method]
        print(f"{method:<25} "
              f"{res['Top-1 Accuracy']:<6.4f}±{res_std['Top-1 Accuracy']:<4.4f} "
              f"{res['Top-5 Accuracy']:<6.4f}±{res_std['Top-5 Accuracy']:<4.4f} "
              f"{res['Coverage']:<7.4f}±{res_std['Coverage']:<6.4f} "
              f"{res['Average Set Size']:<7.4f}±{res_std['Average Set Size']:<6.4f} "
              f"{res['Prob(Size > 1)']:<7.4f}±{res_std['Prob(Size > 1)']:<6.4f} "
              f"{res['Prob(Empty)']:<7.4f}±{res_std['Prob(Empty)']:<6.4f}")

    # --- ConformalPredictor ---
    print("\n--- ConformalPredictor Results ---")
    results_cp = {}
    results_std_cp = {}
    for method in methods:
        avg_results = {}
        std_results = {}
        for metric in ['Top-1 Accuracy', 'Top-5 Accuracy', 'Coverage', 'Average Set Size', 'Prob(Size > 1)', 'Prob(Empty)']:
            values = [run[metric] for run in all_results_cp[method]]
            avg_results[metric] = np.mean(values)
            std_results[metric] = np.std(values)
        results_cp[method] = avg_results
        results_std_cp[method] = std_results

    print(f"\n{'Method':<30} {'Top-1':<15} {'Top-5':<15} {'Coverage':<15} {'Avg Size':<15} {'P(Size>1)':<15} {'P(Empty)':<15}")
    print("-" * 115)
    for method in methods:
        res = results_cp[method]
        res_std = results_std_cp[method]
        print(f"{method:<25} "
              f"{res['Top-1 Accuracy']:<6.4f}±{res_std['Top-1 Accuracy']:<4.4f} "
              f"{res['Top-5 Accuracy']:<6.4f}±{res_std['Top-5 Accuracy']:<4.4f} "
              f"{res['Coverage']:<7.4f}±{res_std['Coverage']:<6.4f} "
              f"{res['Average Set Size']:<7.4f}±{res_std['Average Set Size']:<6.4f} "
              f"{res['Prob(Size > 1)']:<7.4f}±{res_std['Prob(Size > 1)']:<6.4f} "
              f"{res['Prob(Empty)']:<7.4f}±{res_std['Prob(Empty)']:<6.4f}")

    print(f"\n--- Bootstrap 95% Confidence Intervals ({args.bootstrap_samples} bootstrap samples) ---")
    n_bootstrap = args.bootstrap_samples
    
    print("\n-- ConformalModel --")
    for method in methods:
        coverage_runs = [run['Coverage'] for run in all_results_cm[method]]
        size_runs = [run['Average Set Size'] for run in all_results_cm[method]]
        
        coverage_bootstrap = [np.mean(np.random.choice(coverage_runs, size=len(coverage_runs), replace=True)) for _ in range(n_bootstrap)]
        size_bootstrap = [np.mean(np.random.choice(size_runs, size=len(size_runs), replace=True)) for _ in range(n_bootstrap)]

        coverage_ci_lower, coverage_ci_upper = np.percentile(coverage_bootstrap, [2.5, 97.5])
        size_ci_lower, size_ci_upper = np.percentile(size_bootstrap, [2.5, 97.5])
        
        print(f"  {method}:")
        print(f"    Coverage: {results_cm[method]['Coverage']:.4f} [Bootstrap 95% CI: {coverage_ci_lower:.4f}, {coverage_ci_upper:.4f}]")
        print(f"    Avg Size: {results_cm[method]['Average Set Size']:.4f} [Bootstrap 95% CI: {size_ci_lower:.4f}, {size_ci_upper:.4f}]")

    print("\n-- ConformalPredictor --")
    for method in methods:
        coverage_runs = [run['Coverage'] for run in all_results_cp[method]]
        size_runs = [run['Average Set Size'] for run in all_results_cp[method]]
        
        coverage_bootstrap = [np.mean(np.random.choice(coverage_runs, size=len(coverage_runs), replace=True)) for _ in range(n_bootstrap)]
        size_bootstrap = [np.mean(np.random.choice(size_runs, size=len(size_runs), replace=True)) for _ in range(n_bootstrap)]

        coverage_ci_lower, coverage_ci_upper = np.percentile(coverage_bootstrap, [2.5, 97.5])
        size_ci_lower, size_ci_upper = np.percentile(size_bootstrap, [2.5, 97.5])
        
        print(f"  {method}:")
        print(f"    Coverage: {results_cp[method]['Coverage']:.4f} [Bootstrap 95% CI: {coverage_ci_lower:.4f}, {coverage_ci_upper:.4f}]")
        print(f"    Avg Size: {results_cp[method]['Average Set Size']:.4f} [Bootstrap 95% CI: {size_ci_lower:.4f}, {size_ci_upper:.4f}]")

    print(f"\nExperiment completed! Ran {args.repetitions} repetitions for each of {len(methods)} methods.")
    print(f"Statistical reliability: Results averaged across {args.repetitions} different data splits with confidence intervals.")
    print("Complete!")

if __name__ == '__main__':
    main()
