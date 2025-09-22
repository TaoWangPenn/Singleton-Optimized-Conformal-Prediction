import argparse
import time
import numpy as np
import torch
import pickle
from pathlib import Path

from src.conformalizer import ConformalPredictor
from src.utils import validate, split_logits, platt_logits, set_random_seeds
from src.hyperparameter_tuner import HyperparameterTuner


def main():
    """
    Main function to run conformal prediction with pre-computed logits and automatic hyperparameter tuning.
    """
    parser = argparse.ArgumentParser(description='Conformal Prediction with Pre-computed Logits and Auto-tuning')
    parser.add_argument('--logits_file', type=str, default='ImageNet/logits/logits_imgnet_val/resnet152_logits.pkl',
                        help='path to pre-computed logits file')
    parser.add_argument('--num_tune', type=int, default=10000, metavar='NTUNE',
                        help='number of samples for hyperparameter tuning (default: 10000)')
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
    parser.add_argument('--penalty_lambda', type=float, default=None,
                        help='penalty parameter λ (default: auto-tune)')
    parser.add_argument('--use_numba', action='store_true',
                        help='Use Numba for the convex hull method')
    parser.add_argument('--k_reg', type=int, default=None,
                        help='k_reg parameter for RAPS method (default: auto-tune)')
    parser.add_argument('--randomized', action='store_true',
                        help='Use randomized version of RAPS')
    parser.add_argument('--methods', nargs='+', default=['socop'],
                        help='List of conformal methods to run (default: ["socop"])')
    parser.add_argument('--size_list', action='store_true',
                        help='Whether to return the list of sizes for each prediction set')

    args = parser.parse_args()

    # --- Initial Reproducibility Setup ---
    set_random_seeds(args.seed)

    # --- Load and Split Pre-computed Logits ---
    print(f"Loading pre-computed logits from: {args.logits_file}")
    
    logits_path = Path(args.logits_file)
    if not logits_path.exists():
        raise FileNotFoundError(f"Logits file not found: {args.logits_file}")
    
    print("Loading logits...")
    with open(args.logits_file, 'rb') as f:
        full_logits_dataset = pickle.load(f)
    
    print(f"Loaded logits: {full_logits_dataset.tensors[0].shape[0]} samples, {full_logits_dataset.tensors[0].shape[1]} classes")

    # --- Conformal Prediction Methods ---
    print(f"\nAvailable conformal methods: {ConformalPredictor.get_available_methods()}")
    methods = args.methods
    print(f"Selected methods to run: {methods}")
    print(f"Running {args.repetitions} repetitions for each method with different data splits...")
    
    if args.temperature is not None:
        print(f"Using fixed temperature: {args.temperature}")
    else:
        print("Auto-tuning temperature on calibration set for each data split...")
    
    all_results = {} 
    all_timing_results = {} 
    
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
            current_seed = args.seed + run * 100
            set_random_seeds(current_seed)
            
            # Split data into three sets for hyperparameter tuning
            tune_dataset, calib_dataset, val_dataset = split_logits(
                full_logits_dataset,
                num_tune=args.num_tune,
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

            # --- Hyperparameter Tuning ---
            if HyperparameterTuner.needs_tuning(method):
                tuner = HyperparameterTuner(args.alpha, temperature, verbose=False, randomized=args.randomized, use_numba=args.use_numba)
                tuned_params = tuner.get_tuned_parameters(method, tune_dataset, args)
            else:
                tuned_params = {}

            # --- Create ConformalPredictor ---
            predictor_kwargs = {
                'calib_logits': calib_dataset,
                'alpha': args.alpha,
                'method': method,
                'temperature': temperature,
                'verbose': False,
            }
            
            # Add method-specific parameters
            if method == 'raps':
                predictor_kwargs.update(tuned_params)
                predictor_kwargs['randomized'] = args.randomized
            elif method == 'socop':
                predictor_kwargs.update(tuned_params)
                predictor_kwargs['use_numba'] = args.use_numba
            
            conformal_predictor = ConformalPredictor(**predictor_kwargs)
            
            # --- Validation ---
            if args.size_list:
                top1, top5, coverage, avg_size, prob_gt1, prob_empty, size_list = validate(
                    val_loader=val_loader,
                    model_or_predictor=conformal_predictor,
                    val_logits=val_dataset,
                    print_bool=False,
                    return_size_list=True 
                )
                Path('size_list').mkdir(exist_ok=True)
                np.save(f'size_list/size_list_{method}_seed{current_seed}.npy', np.array(size_list))
            else:
                top1, top5, coverage, avg_size, prob_gt1, prob_empty = validate(
                    val_loader=val_loader,
                    model_or_predictor=conformal_predictor,
                    val_logits=val_dataset,
                    print_bool=False,
                    return_size_list=False
                )

            elapsed_time = time.time() - start_time
            
            # Store results for this run
            run_results = {
                'Top-1 Accuracy': top1,
                'Top-5 Accuracy': top5,
                'Coverage': coverage,
                'Average Set Size': avg_size,
                'Prob(Size > 1)': prob_gt1,
                'Prob(Empty)': prob_empty
            }
            
            method_results.append(run_results)
            method_times.append(elapsed_time)
            
            # Print parameters information if method supports tuning
            if tuned_params:
                print(f"Run {run + 1} completed in {elapsed_time:.3f}s - Coverage: {coverage:.4f}, Avg Size: {avg_size:.4f}, P(size > 1): {prob_gt1:.4f}, P(empty): {prob_empty:.4f}, Params: {tuned_params}")
            else:
                print(f"Run {run + 1} completed in {elapsed_time:.3f}s - Coverage: {coverage:.4f}, Avg Size: {avg_size:.4f}, P(size > 1): {prob_gt1:.4f}, P(empty): {prob_empty:.4f}")
        
        all_results[method] = method_results
        all_timing_results[method] = method_times

    results = {}
    timing_results = {}
    results_std = {}
    timing_std = {}
    
    for method in methods:
        avg_results = {}
        std_results = {}
        for metric in ['Top-1 Accuracy', 'Top-5 Accuracy', 'Coverage', 'Average Set Size', 'Prob(Size > 1)', 'Prob(Empty)']:
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
    
    print(f"\n{'Method':<30} {'Top-1':<15} {'Top-5':<15} {'Coverage':<15} {'Avg Size':<15} {'P(Size>1)':<15} {'P(Empty)':<15} {'Time (s)':<10}")
    print("-" * 115)
    
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
              f"{res['Prob(Empty)']:<7.4f}±{res_std['Prob(Empty)']:<6.4f} "
              f"{time_taken:<6.3f}±{time_std:<4.3f}")
    
    # Total experiment time
    total_time = sum(timing_results.values()) * args.repetitions
    print(f"\nTotal Experiment Time: {total_time:.3f}s")
    
    print(f"\nExperiment completed! Ran {args.repetitions} repetitions for each of {len(methods)} methods.")
    print(f"Statistical reliability: Results averaged across {args.repetitions} different data splits.")
    print("Complete!")


if __name__ == '__main__':
    main()
