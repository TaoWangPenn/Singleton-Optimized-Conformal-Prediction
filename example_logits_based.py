import argparse
import time
import random
import numpy as np
import torch
import pickle
from pathlib import Path

from src.conformalizer import ConformalPredictor
from src.utils import validate, split_logits, platt_logits, set_random_seeds


def main():
    """
    Main function to run conformal prediction with pre-computed logits.
    """
    parser = argparse.ArgumentParser(description='Conformal Prediction with Pre-computed Logits')
    parser.add_argument('--logits_file', type=str, default='models/logits/resnet152_logits.pkl', 
                        help='path to pre-computed logits file (default: models/logits/resnet152_logits.pkl)')
    parser.add_argument('--num_calib', type=int, default=20000, metavar='NCALIB', 
                        help='number of calibration points (default: 20000)')
    parser.add_argument('--num_val', type=int, default=20000, metavar='NVAL', 
                        help='number of validation points (default: 20000)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BSZ', 
                        help='batch size for validation (default: 64)')
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

    args = parser.parse_args()

    # --- Initial Reproducibility Setup ---
    set_random_seeds(args.seed)

    # --- Device Configuration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load and Split Pre-computed Logits ---
    print(f"Loading pre-computed logits from: {args.logits_file}")
    
    # Extract model name from file path for the utility function
    logits_path = Path(args.logits_file)
    if not logits_path.exists():
        raise FileNotFoundError(f"Logits file not found: {args.logits_file}")
    
    # Load the logits directly since we have the full path
    print("Loading logits...")
    with open(args.logits_file, 'rb') as f:
        full_logits_dataset = pickle.load(f)
    
    print(f"Loaded logits: {full_logits_dataset.tensors[0].shape[0]} samples, {full_logits_dataset.tensors[0].shape[1]} classes")

    # --- Conformal Prediction Methods ---
    print(f"\nAvailable conformal methods: {ConformalPredictor.get_available_methods()}")
    methods = [
        'naive', 
        'length-optimization',
        'length-optimization-weighted',
        'm-criterion-simple', 
        'm-criterion-weighted', 
        'm-criterion-randomized', 
        'length-optimization-randomized'
    ]
    
    print(f"Running {args.repetitions} repetitions for each method with different data splits...")
    
    if args.temperature is not None:
        print(f"Using fixed temperature: {args.temperature}")
    else:
        print("Auto-tuning temperature on calibration set for each data split...")
    
    # Store results for comparison - now with multiple runs
    all_results = {}  # method -> list of runs
    all_timing_results = {}  # method -> list of times
    
    # Initialize storage for all methods
    for method in methods:
        all_results[method] = []
        all_timing_results[method] = []

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Running method: {method} ({args.repetitions} repetitions)")
        print(f"{'='*80}")
        
        method_results = []
        method_times = []
        
        for run in range(args.repetitions):
            print(f"\n--- Run {run + 1}/{args.repetitions} for {method} ---")
            
            start_time = time.time()
            
            # Set different random seed for each run to get different data splits
            current_seed = args.seed + run * 10
            set_random_seeds(current_seed)
            
            # Split data with new random seed
            calib_dataset, val_dataset = split_logits(
                full_logits_dataset, 
                num_calib=args.num_calib, 
                num_val=args.num_val
            )

            # Create data loaders for this run
            calib_loader = torch.utils.data.DataLoader(
                calib_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
            )

            # --- Temperature Scaling ---
            if args.temperature is not None:
                temperature = args.temperature
            else:
                T_param = platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01)
                temperature = T_param.item()
            
            # --- Create ConformalPredictor ---
            conformal_predictor = ConformalPredictor(
                calib_logits=calib_dataset,
                alpha=args.alpha,
                method=method,
                temperature=temperature,
                verbose=False
            )
            
            print(f"Evaluating over validation set (run {run + 1})...")
            
            # --- Validation ---
            top1, top5, coverage, avg_size, prob_gt1 = validate(
                val_loader=val_loader,  # This is used for batch_size only
                model_or_predictor=conformal_predictor,
                val_logits=val_dataset,  # Pass the validation logits
                print_bool=False
            )
            
            elapsed_time = time.time() - start_time
            
            # Store results for this run
            run_results = {
                'Top-1 Accuracy': top1,
                'Top-5 Accuracy': top5,
                'Coverage': coverage,
                'Average Set Size': avg_size,
                'Prob(Size > 1)': prob_gt1
            }
            
            method_results.append(run_results)
            method_times.append(elapsed_time)
            
            print(f"Run {run + 1} completed in {elapsed_time:.3f}s - Coverage: {coverage:.4f}, Avg Size: {avg_size:.4f}")
        
        # Store all runs for this method
        all_results[method] = method_results
        all_timing_results[method] = method_times
        
        # Calculate and display statistics for this method
        avg_results = {}
        std_results = {}
        for metric in ['Top-1 Accuracy', 'Top-5 Accuracy', 'Coverage', 'Average Set Size', 'Prob(Size > 1)']:
            values = [run[metric] for run in method_results]
            avg_results[metric] = np.mean(values)
            std_results[metric] = np.std(values)
        
        avg_time = np.mean(method_times)
        std_time = np.std(method_times)
        
        print(f"\n{method} Summary ({args.repetitions} different data splits):")
        print(f"  Coverage: {avg_results['Coverage']:.4f} ± {std_results['Coverage']:.4f}")
        print(f"  Avg Size: {avg_results['Average Set Size']:.4f} ± {std_results['Average Set Size']:.4f}")
        print(f"  Time: {avg_time:.3f} ± {std_time:.3f} seconds")

    # Calculate final averaged results for comparison table
    results = {}
    timing_results = {}
    results_std = {}
    timing_std = {}
    
    for method in methods:
        # Calculate mean across all runs
        avg_results = {}
        std_results = {}
        for metric in ['Top-1 Accuracy', 'Top-5 Accuracy', 'Coverage', 'Average Set Size', 'Prob(Size > 1)']:
            values = [run[metric] for run in all_results[method]]
            avg_results[metric] = np.mean(values)
            std_results[metric] = np.std(values)
        
        results[method] = avg_results
        results_std[method] = std_results
        timing_results[method] = np.mean(all_timing_results[method])
        timing_std[method] = np.std(all_timing_results[method])

    # --- Print Final Comparison ---
    print("\n" + "="*100)
    print(f"FINAL COMPARISON OF ALL METHODS (Average across {args.repetitions} different data splits)")
    print("="*100)
    
    print(f"\n{'Method':<30} {'Top-1':<15} {'Top-5':<15} {'Coverage':<15} {'Avg Size':<15} {'P(Size>1)':<15} {'Time (s)':<10}")
    print("-" * 100)
    
    for method in methods:
        res = results[method]
        res_std = results_std[method]
        time_taken = timing_results[method]
        time_std = timing_std[method]
        
        print(f"{method:<25} "
              f"{res['Top-1 Accuracy']:<6.4f}±{res_std['Top-1 Accuracy']:<4.4f} "
              f"{res['Top-5 Accuracy']:<6.4f}±{res_std['Top-5 Accuracy']:<4.4f} "
              f"{res['Coverage']:<7.4f}±{res_std['Coverage']:<6.4f} "
              f"{res['Average Set Size']:<7.4f}±{res_std['Average Set Size']:<6.4f} "
              f"{res['Prob(Size > 1)']:<7.4f}±{res_std['Prob(Size > 1)']:<6.4f} "
              f"{time_taken:<6.3f}±{time_std:<4.3f}")
    
    # Total experiment time
    total_time = sum(timing_results.values()) * args.repetitions
    print(f"\nTotal Experiment Time: {total_time:.3f}s")
    
    # Bootstrap confidence intervals for more robust statistics
    print(f"\n--- Bootstrap 95% Confidence Intervals ({args.bootstrap_samples} bootstrap samples) ---")
    n_bootstrap = args.bootstrap_samples
    
    for method in methods:
        # Extract all runs for this method
        coverage_runs = [run['Coverage'] for run in all_results[method]]
        size_runs = [run['Average Set Size'] for run in all_results[method]]
        
        # Bootstrap resampling for coverage
        coverage_bootstrap = []
        size_bootstrap = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample_coverage = np.random.choice(coverage_runs, size=len(coverage_runs), replace=True)
            bootstrap_sample_size = np.random.choice(size_runs, size=len(size_runs), replace=True)
            
            coverage_bootstrap.append(np.mean(bootstrap_sample_coverage))
            size_bootstrap.append(np.mean(bootstrap_sample_size))
        
        # Calculate percentile-based confidence intervals
        coverage_ci_lower = np.percentile(coverage_bootstrap, 2.5)
        coverage_ci_upper = np.percentile(coverage_bootstrap, 97.5)
        size_ci_lower = np.percentile(size_bootstrap, 2.5)
        size_ci_upper = np.percentile(size_bootstrap, 97.5)
        
        print(f"  {method}:")
        print(f"    Coverage: {results[method]['Coverage']:.4f} [Bootstrap 95% CI: {coverage_ci_lower:.4f}, {coverage_ci_upper:.4f}]")
        print(f"    Avg Size: {results[method]['Average Set Size']:.4f} [Bootstrap 95% CI: {size_ci_lower:.4f}, {size_ci_upper:.4f}]")
    
    print(f"\nExperiment completed! Ran {args.repetitions} repetitions for each of {len(methods)} methods.")
    print(f"Statistical reliability: Results averaged across {args.repetitions} different data splits with confidence intervals.")
    print("Complete!")


if __name__ == '__main__':
    main()
