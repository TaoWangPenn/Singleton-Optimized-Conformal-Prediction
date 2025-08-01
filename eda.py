import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet152_Weights
import numpy as np
import random
import torch.backends.cudnn as cudnn
import time 


from src.optimization.M_criterion import (
    OptimizationProblem,
    SLSQPSolver,
    PGASolver,
    find_best_solution_with_multistart,
    conjecture_gamma,
    print_results
)

def main():
    """
    Main function to run the conformal prediction example.
    """
    parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet (Updated 2025)')
    parser.add_argument('--data', default='/shared_data0/tawan/Imagenet_val_reorganized', metavar='/shared_data0/tawan/Imagenet_val_reorganized', help='path to Imagenet validation set')
    parser.add_argument('--batch_size', type=int, default=64, metavar='BSZ', help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='NW', help='number of workers for data loading (default: 4)')
    parser.add_argument('--num_calib', type=int, default=20000, metavar='NCALIB', help='number of calibration points (default: 20000)')
    parser.add_argument('--num_val', type=int, default=10000, metavar='NVAL', help='number of validation points (default: 10000)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--alpha', type=float, default=0.05, metavar='A', help='error rate (default: 0.05)')

    args = parser.parse_args()

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- Device Configuration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preprocessing ---
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the full dataset
    full_dataset = torchvision.datasets.ImageFolder(args.data, transform)
    
    # Dynamically get dataset size
    full_dataset = torchvision.datasets.ImageFolder(args.data, transform)
    
    # Dynamically get dataset size
    dataset_size = len(full_dataset)
    if args.num_calib + args.num_val > dataset_size:
        raise ValueError(f"Requested {args.num_calib} calibration and {args.num_val} validation points, but dataset only has {dataset_size} samples.")

    # Split dataset into three parts: calibration, validation, and unused
    num_unused = dataset_size - args.num_calib - args.num_val
    calib_dataset, val_dataset, _ = torch.utils.data.random_split(
        full_dataset, [args.num_calib, args.num_val, num_unused]
    )

    calib_loader = torch.utils.data.DataLoader(
        calib_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # --- Model Loading ---
    print("Loading ResNet-152 model...")
    weights = ResNet152_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet152(weights=weights).to(device)
    model.eval()

    # --- Pre-Conformalization Analysis ---
    print("\n--- Starting Pre-Conformalization Analysis on the Calibration Set ---")
    
    # Step 1: Perform Platt Scaling to get the temperature T
    from src.utils import get_logits_targets, platt_logits, accuracy, AverageMeter
    import torch.utils.data as tdata
    from scipy.special import softmax

    print("\n1. Performing Platt Scaling...")
    calib_logits_dataset = get_logits_targets(model, calib_loader)
    logits_loader = tdata.DataLoader(calib_logits_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    T = platt_logits(logits_loader)
    print(f"   Optimal Temperature T = {T.item():.4f}")

    # Step 2: Calculate accuracy and collect max probabilities on the calibration set
    print("\n2. Analyzing model accuracy and confidence distribution...")
    top1 = AverageMeter('top1')
    top5 = AverageMeter('top5')
    all_max_probs = []

    # Define threshold before the loop
    threshold = (1 - args.alpha) / 2
    found_first = False  

    with torch.no_grad():
        for logits, targets in logits_loader:
            scaled_logits = logits.to(device) / T
            
            # Calculate accuracy
            prec1, prec5 = accuracy(scaled_logits, targets.to(device), topk=(1, 5))
            top1.update(prec1.item(), n=logits.shape[0])
            top5.update(prec5.item(), n=logits.shape[0])

            # Calculate and collect max probabilities
            probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()
            all_max_probs.extend(np.max(probs, axis=1))

            # Find and print only the very first sample with max_prob below threshold
            if not found_first:
                for i, prob in enumerate(probs):
                    if np.max(prob) <= threshold:
                        print("\n--- First Sample with max_p_hat <= Threshold ---")
                        
                        prediction_index = np.argmax(prob)
                        is_correct = (prediction_index == targets[i].item())
                        sorted_probs = np.sort(prob)[::-1]
                        
                        # Check top-1 and top-5 correctness
                        top5_indices = np.argsort(prob)[::-1][:5]
                        is_top1_correct = (prediction_index == targets[i].item())
                        is_top5_correct = (targets[i].item() in top5_indices)

                        print(f"Index in batch: {i}")
                        print(f"True Label: {targets[i].item()}, Model Prediction: {prediction_index}")
                        print(f"Classification Correct: {is_correct}")
                        print(f"Max Probability: {np.max(prob):.4f}")
                        print(f"Top 50 Probabilities (sorted): {np.round(sorted_probs[:50], 4)}")
                        print(f"Classification Correct (Top-1): {is_top1_correct}")
                        print(f"Classification Correct (Top-5): {is_top5_correct}")

                        # --- Start of Simplified M-criterion Analysis ---
                        print("\n--- Simplified M-criterion Analysis (Full Dimension) ---")
                        
                        p_sample_indices = np.argsort(prob)[::-1]
                        p_sample = prob[p_sample_indices]
                        gamma_conjecture_sorted = conjecture_gamma(p_sample, args.alpha)
                        gamma_conjecture = np.zeros_like(gamma_conjecture_sorted)
                        gamma_conjecture[p_sample_indices] = gamma_conjecture_sorted

                        # 1. Define a full-dimensional optimization problem (threshold=0)
                        print("\nCreating full-dimensional optimization problem (threshold=0)...")
                        problem_full = OptimizationProblem(prob, args.alpha, threshold=None)
                        
                        # 2. Calculate the conjectured gamma to use as the initial point
                        print("Calculating conjectured gamma as initial point...")
                        gamma0 = gamma_conjecture

                        # 3. Solve with SLSQP starting from the conjecture
                        slsqp_solver = SLSQPSolver(problem_full, tol=1e-9, max_iter=1000)
                        start_time_slsqp = time.time()
                        result_slsqp = slsqp_solver.solve(gamma0=gamma0)
                        end_time_slsqp = time.time()
                        print_results("SLSQP (from conjecture)", result_slsqp, end_time_slsqp - start_time_slsqp)

                        # 4. Solve with PGA starting from the conjecture
                        pga_solver = PGASolver(problem_full, tol=1e-9, max_iter=1000)
                        start_time_pga = time.time()
                        result_pga = pga_solver.solve(gamma0=gamma0)
                        end_time_pga = time.time()
                        print_results("PGA (from conjecture)", result_pga, end_time_pga - start_time_pga)

                        # 5. Explicitly evaluate and print the conjecture's objective value for comparison
                        print("\n" + "="*50)
                        print("--- Conjectured Gamma Evaluation (Initial Point) ---")
                        log_g_conj = problem_full.objective(gamma0)
                        g_conj = np.exp(log_g_conj)
                        print(f"Initial Point Value (log_g from conjecture): {log_g_conj:.6f}")
                        print(f"Initial Point Value (g from conjecture):     {g_conj:.6f}")
                        print("="*50)

                        print("--- End of Simplified Analysis ---\n")
                        # --- End of M-criterion Analysis ---

                        print("--- End of Sample ---\n")
                        found_first = True  # Set the flag to True after printing
                        break

    print(f"\n   Model Accuracy on Calibration Set ({len(calib_dataset)} samples):")
    print(f"   - Top-1 Accuracy: {top1.avg:.2f}%")
    print(f"   - Top-5 Accuracy: {top5.avg:.2f}%")

    # Step 3: Perform detailed analysis on the distribution of max probabilities
    print("\n3. Analyzing confidence (max_p_hat) distribution...")
    all_max_probs = np.array(all_max_probs)
    threshold = (1 - args.alpha) / 2
    
    below_threshold_count = np.sum(all_max_probs <= threshold)
    total_count = len(all_max_probs)
    below_threshold_percent = (below_threshold_count / total_count) * 100
    
    print(f"   - Total calibration samples: {total_count}")
    print(f"   - Confidence threshold for triggering SLSQP: {threshold:.4f}")
    print(f"   - Samples with max_p_hat <= threshold: {below_threshold_count} ({below_threshold_percent:.2f}%)")
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    p_values = np.percentile(all_max_probs, percentiles)
    print("\n   Confidence Percentiles (max_p_hat):")
    for p, val in zip(percentiles, p_values):
        print(f"   - {p:2d}th percentile: {val:.4f}")
    print("--- End of Pre-Analysis ---\n")

    print("\nComplete!")

if __name__ == '__main__':
    main()
