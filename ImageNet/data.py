import os
import pickle
import torch
import torch.utils.data as tdata
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from src.utils import get_logits_targets
from .model import load_pretrained_model

def get_model_specific_transforms(model_name: str = None):
    """
    Get model-specific transforms based on official pretrained weights.
    
    Args:
        model_name: Name of the model (e.g., 'resnet152', 'efficientnet_v2_l')
                   If None, returns generic ImageNet transforms
    
    Returns:
        Model-specific transform pipeline
    """
    if model_name is None:
        return get_imagenet_transforms()
    
    weight_transforms_map = {
        'resnet152v1': lambda: torchvision.models.ResNet152_Weights.IMAGENET1K_V1.transforms(), # replicate RAPS
        'resnet152': lambda: torchvision.models.ResNet152_Weights.IMAGENET1K_V2.transforms(),
        'efficientnet_v2_l': lambda: torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms(),
        'convnext_base': lambda: torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms(),
        'swin_v2_b': lambda: torchvision.models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms(),
        'vit_h_14': lambda: torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms(),
    }
    
    if model_name in weight_transforms_map:
        try:
            official_transform = weight_transforms_map[model_name]()
            print(f"Using official transforms for {model_name}: {official_transform}")
            return official_transform
        except Exception as e:
            print(f"Warning: Could not load official transforms for {model_name}: {e}")
            print("Falling back to generic ImageNet transforms")
            return get_imagenet_transforms()
    else:
        print(f"Warning: No specific transforms found for {model_name}. Using generic ImageNet transforms.")
        return get_imagenet_transforms()


def get_imagenet_transforms():
    """Get standard ImageNet preprocessing transforms."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return transform


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def load_dataset(
    data_path: str, 
    batch_size: int = 64, 
    num_workers: int = 4,
    model_name: str = None
) -> tdata.DataLoader:
    """
    Load ImageNet validation dataset with model-specific transforms.
    
    Args:
        data_path: Path to dataset
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        model_name: Name of the model to get specific transforms for
    """
    transform = get_model_specific_transforms(model_name)
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transform)
    
    loader = tdata.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Loaded dataset: {len(dataset)} samples")
    if model_name:
        print(f"Using model-specific transforms for: {model_name}")
    
    return loader


# ============================================================================
# Logits Pre-computation Functions
# ============================================================================

def precompute_logits(
    model: torch.nn.Module, 
    data_loader: tdata.DataLoader, 
    model_name: str,
    save_dir: str = './logits_cache'
) -> str:
    """
    Pre-compute logits for a model and save them to disk.
    """
    print(f"Pre-computing logits for {model_name}...")
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logits_dataset = get_logits_targets(model, data_loader)
    
    logits_file = save_path / f"{model_name}_logits.pkl"
    with open(logits_file, 'wb') as f:
        pickle.dump(logits_dataset, f)
    
    print(f"Saved logits to: {logits_file}")
    print(f"Logits shape: {logits_dataset.tensors[0].shape}")
    
    return str(logits_file)


def precompute_multiple_models(
    model_names: List[str],
    data_path: str,
    save_dir: str = './logits_cache',
    batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, str]:
    """
    Pre-compute logits for multiple models using model-specific transforms.
    """
    saved_files = {}
    
    for model_name in model_names:
        try:
            print(f"\n=== Processing {model_name} ===")
            data_loader = load_dataset(data_path, batch_size, num_workers, model_name)
            model = load_pretrained_model(model_name)
            
            logits_file = precompute_logits(model, data_loader, model_name, save_dir)
            saved_files[model_name] = logits_file
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue
    
    print(f"\nSuccessfully processed {len(saved_files)} models:")
    for model_name, file_path in saved_files.items():
        print(f"  ✓ {model_name}: {file_path}")
    
    return saved_files
