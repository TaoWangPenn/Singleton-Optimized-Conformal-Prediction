#!/usr/bin/env python3
"""
General script to pre-compute logits for any available model.

This script can pre-compute logits for any supported model and save them
to the models/logits directory. ResNet152 is used as the default.

Usage:
    python precompute_logits.py --model resnet152 --data_path /path/to/imagenet/val
    python precompute_logits.py --list_models  # List all available models
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ImageNet.data import precompute_logits, load_dataset
from ImageNet.model import load_pretrained_model, get_available_models


def main():
    parser = argparse.ArgumentParser(description='Pre-compute logits for any model')
    parser.add_argument(
        '--model', 
        type=str, 
        default='resnet152',
        help='Model name to use (default: resnet152). Use --list_models to see available models.'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='/path/to/imagenet/val',
        help='Path to ImageNet validation dataset (default: /path/to/imagenet/val)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help='Batch size for data loading (default: 64)'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of worker processes for data loading (default: 4)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./models/logits',
        help='Directory to save logits (default: ./models/logits)'
    )
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List all available models and exit'
    )
    
    args = parser.parse_args()
    
    # List available models and exit
    if args.list_models:
        available_models = get_available_models()
        print("Available models:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}. {model}")
        return 0
    
    # Validate model name
    available_models = get_available_models()
    if args.model not in available_models:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(available_models)}")
        print("Use --list_models to see all available models.")
        return 1
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist.")
        print("Please provide a valid path to the ImageNet validation dataset.")
        print(f"Example: python precompute_logits.py --model {args.model} --data_path /path/to/imagenet/val")
        return 1
    
    print("=" * 60)
    print(f"Pre-computing logits for {args.model}")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 60)
    
    try:
        # Load the dataset
        print("\n1. Loading dataset...")
        data_loader = load_dataset(
            args.data_path, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            model_name=args.model
        )
        
        # Load the model
        print(f"\n2. Loading {args.model} model...")
        model = load_pretrained_model(args.model)
        
        # Pre-compute and save logits
        print(f"\n3. Pre-computing logits...")
        logits_file = precompute_logits(
            model, 
            data_loader, 
            args.model, 
            save_dir=args.save_dir
        )
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print(f"Logits saved to: {logits_file}")
        print("=" * 60)
        
        # Show file size
        file_size = Path(logits_file).stat().st_size / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
