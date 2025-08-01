import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet152_Weights
import numpy as np
import random
import torch.backends.cudnn as cudnn
import time

# Correctly import from your project structure
from src.conformalizer import ConformalModel, ConformalPredictor
from src.utils import validate, get_logits_targets, platt_scale


def main():
    """
    Main function to run the conformal prediction example.
    """
    parser = argparse.ArgumentParser(description='Conformalize Torchvision Model on Imagenet (Updated 2025)')
    parser.add_argument('data', metavar='/shared_data0/tawan/Imagenet_val_reorganized', help='path to Imagenet validation set')
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
    # Modern transform for ImageNet validation
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
    # Use the latest recommended way to load pretrained models (as of 2025)
    print("Loading ResNet-152 model...")
    weights = ResNet152_Weights.IMAGENET1K_V2 # Using V2 for better accuracy
    model = torchvision.models.resnet152(weights=weights).to(device)
    model.eval()

    # --- Platt Scaling (once for all methods) ---
    from src.utils import platt_scale
    T, calib_logits = platt_scale(model, calib_loader)

    # --- Pre-compute validation logits for ConformalPredictor (once for all methods) ---
    print("Pre-computing validation logits for ConformalPredictor...")
    val_logits = get_logits_targets(model, val_loader)

    # --- Conformalization and Validation ---
    print(f"\nAvailable conformal methods: {ConformalModel.get_available_methods()}")
    methods = ['naive', 'length-optimization', 'length-optimization-weighted', 'm-criterion-simple', 'm-criterion-weighted', 'm-criterion-randomized', 'length-optimization-randomized']
    
    # Store results for comparison
    conformal_model_results = {}
    conformal_predictor_results = {}
    timing_results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running method: {method}")
        print(f"{'='*60}")
        
        # =================================================================
        # APPROACH 1: ConformalModel (High-level wrapper)
        # =================================================================
        print(f"\n--- APPROACH 1: ConformalModel (High-level wrapper) ---")
        
        start_time = time.time()
        
        # --- Conformalization ---
        print("Conformalizing the model with ConformalModel...")
        conformal_model = ConformalModel(
            model=model, 
            calib_loader=calib_loader, 
            alpha=args.alpha, 
            method=method, 
            T=T, 
            calib_logits=calib_logits,
            verbose=False  # Set to False to reduce output clutter
        )

        print("Model calibrated and conformalized!")
        
        # Show method information
        method_info = conformal_model.get_method_info()
        print(f"Method info: {method_info}")
        
        print("Now evaluating over the validation set...")
        
        # --- Validation ---
        top1_cm, top5_cm, coverage_cm, size_cm, prob_gt1_cm = validate(val_loader, conformal_model, print_bool=False)
        
        conformal_model_time = time.time() - start_time
        
        conformal_model_results[method] = {
            'Top-1 Accuracy': top1_cm,
            'Top-5 Accuracy': top5_cm,
            'Coverage': coverage_cm,
            'Average Set Size': size_cm,
            'Prob(Size > 1)': prob_gt1_cm
        }
        
        print(f"ConformalModel completed in {conformal_model_time:.3f} seconds")
        
        # =================================================================
        # APPROACH 2: ConformalPredictor (Low-level, logits-based)
        # =================================================================
        print(f"\n--- APPROACH 2: ConformalPredictor (Low-level, logits-based) ---")
        
        start_time = time.time()
        
        # --- Conformalization ---
        print("Creating ConformalPredictor...")
        conformal_predictor = ConformalPredictor(
            calib_logits=calib_logits,  # Pass TensorDataset directly
            alpha=args.alpha,
            method=method,
            temperature=T.item(),
            verbose=False
        )
        
        print("ConformalPredictor created and calibrated!")
        
        # Show method information
        method_info_cp = conformal_predictor.get_method_info()
        print(f"Method info: {method_info_cp}")
        
        print("Now evaluating over the validation set...")
        
        # --- Validation with pre-computed logits ---
        top1_cp, top5_cp, coverage_cp, size_cp, prob_gt1_cp = validate(
            val_loader, conformal_predictor, val_logits=val_logits, print_bool=False
        )
        
        conformal_predictor_time = time.time() - start_time
        
        conformal_predictor_results[method] = {
            'Top-1 Accuracy': top1_cp,
            'Top-5 Accuracy': top5_cp,
            'Coverage': coverage_cp,
            'Average Set Size': size_cp,
            'Prob(Size > 1)': prob_gt1_cp
        }
        
        print(f"ConformalPredictor completed in {conformal_predictor_time:.3f} seconds")
        
        # Store timing comparison
        timing_results[method] = {
            'ConformalModel': conformal_model_time,
            'ConformalPredictor': conformal_predictor_time,
            'Speedup': conformal_model_time / conformal_predictor_time if conformal_predictor_time > 0 else float('inf')
        }
        
        # --- Results Comparison for this method ---
        print(f"\n--- Results Comparison for {method} ---")
        print(f"{'Metric':<20} {'ConformalModel':<15} {'ConformalPredictor':<18} {'Difference':<12}")
        print("-" * 70)
        for metric in ['Top-1 Accuracy', 'Top-5 Accuracy', 'Coverage', 'Average Set Size', 'Prob(Size > 1)']:
            cm_val = conformal_model_results[method][metric]
            cp_val = conformal_predictor_results[method][metric]
            diff = abs(cm_val - cp_val)
            print(f"{metric:<20} {cm_val:<15.4f} {cp_val:<18.4f} {diff:<12.6f}")
        
        print(f"\nTiming: ConformalModel: {conformal_model_time:.3f}s, ConformalPredictor: {conformal_predictor_time:.3f}s")
        print(f"Speedup: {timing_results[method]['Speedup']:.2f}x")
        
        print(f"--- Finished method: {method} ---\n")

    # --- Print Final Comparison ---
    print("\n" + "="*80)
    print("FINAL COMPARISON OF ALL METHODS")
    print("="*80)
    
    print("\n--- ConformalModel Results ---")
    for method, res in conformal_model_results.items():
        print(f"\nMethod: {method}")
        for metric, value in res.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n--- ConformalPredictor Results ---")
    for method, res in conformal_predictor_results.items():
        print(f"\nMethod: {method}")
        for metric, value in res.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n--- Timing Comparison ---")
    print(f"{'Method':<25} {'ConformalModel':<15} {'ConformalPredictor':<18} {'Speedup':<10}")
    print("-" * 75)
    for method, timing in timing_results.items():
        print(f"{method:<25} {timing['ConformalModel']:<15.3f} {timing['ConformalPredictor']:<18.3f} {timing['Speedup']:<10.2f}x")
    
    # Calculate average speedup
    avg_speedup = np.mean([timing['Speedup'] for timing in timing_results.values()])
    print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    print("\nComplete!")

if __name__ == '__main__':
    main()
