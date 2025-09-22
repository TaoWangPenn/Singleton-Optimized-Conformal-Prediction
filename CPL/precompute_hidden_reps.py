#!/usr/bin/env python3
"""
Script to pre-compute last hidden layer representations for any available vision model.

This script extracts the inputs to the final linear layer (i.e., the features
or hidden representations) and saves them along with their labels.

Usage:
    python CPL/precompute_hidden_reps.py --model convnext_base --data_path /path/to/imagenet/val
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as tdata

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ImageNet.data import load_dataset
from ImageNet.model import load_pretrained_model, get_available_models

def precompute_hidden_reps(model, data_loader, model_name, save_dir):
    """
    Extracts hidden representations from the penultimate layer of a model and saves them.

    Args:
        model (nn.Module): The model from which to extract features.
        data_loader (DataLoader): DataLoader for the dataset.
        model_name (str): Name of the model, used for saving the file.
        save_dir (str): Directory to save the output file.

    Returns:
        str: The path to the saved file.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    hidden_reps_list = []
    labels_list = []

    def hook(module, input, output):
        hidden_reps_list.append(input[0].detach().cpu())

    final_layer = None
    if hasattr(model, 'fc'):
        final_layer = model.fc
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for layer in reversed(model.classifier):
                if isinstance(layer, nn.Linear):
                    final_layer = layer
                    break
        elif isinstance(model.classifier, nn.Linear):
            final_layer = model.classifier
    elif hasattr(model, 'head'):
        final_layer = model.head
    elif hasattr(model, 'heads'):
        if isinstance(model.heads, nn.Sequential):
            for layer in reversed(model.heads):
                if isinstance(layer, nn.Linear):
                    final_layer = layer
                    break
        elif isinstance(model.heads, nn.Linear):
            final_layer = model.heads
    
    if final_layer is None:
        raise TypeError(f"Could not automatically find the final linear layer for model '{model_name}'.")

    handle = final_layer.register_forward_hook(hook)

    print("Extracting hidden representations...")
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            model(images)
            labels_list.append(labels.cpu())

    handle.remove()

    all_reps = torch.cat(hidden_reps_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    hidden_reps_dataset = tdata.TensorDataset(all_reps, all_labels)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    output_file = save_path / f"{model_name}_hidden_reps.pkl"

    print(f"Saving hidden representations to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(hidden_reps_dataset, f)

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Pre-compute hidden representations for a vision model.')
    parser.add_argument(
        '--model', 
        type=str, 
        default='resnet152',
        help='Model name to use. Use --list_models to see options.'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='/path/to/imagenet/val',
        help='Path to ImageNet validation dataset.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help='Batch size for data loading.'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4,
        help='Number of worker processes for data loading.'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='/path/to/your/CPL_HiddenRep/img_val',
        help='Directory to save hidden representations.'
    )
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List all available models and exit.'
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for i, model in enumerate(get_available_models(), 1):
            print(f"  {i:2d}. {model}")
        return 0
    
    if args.model not in get_available_models():
        print(f"Error: Unknown model '{args.model}'. Use --list_models to see available models.")
        return 1
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist.")
        return 1
    
    print("=" * 60)
    print(f"Pre-computing hidden representations for {args.model}")
    print(f"Target directory: {args.save_dir}")
    print("=" * 60)
    
    try:
        print("\n1. Loading dataset...")
        data_loader = load_dataset(
            args.data_path, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            model_name=args.model
        )
        
        print(f"\n2. Loading {args.model} model...")
        model = load_pretrained_model(args.model)
        
        print(f"\n3. Pre-computing hidden representations...")
        output_file = precompute_hidden_reps(
            model, 
            data_loader, 
            args.model, 
            save_dir=args.save_dir
        )
        
        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print(f"Hidden representations saved to: {output_file}")
        print("=" * 60)
        
        file_size = Path(output_file).stat().st_size / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
