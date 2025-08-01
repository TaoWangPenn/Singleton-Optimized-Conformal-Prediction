import torch
import torchvision
from typing import Dict, List, Optional, Tuple, Union


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_pretrained_model(model_name: str) -> torch.nn.Module:
    """
    Load a pre-trained model by name.
    
    Args:
        model_name: Name of the model (e.g., 'resnet50', 'efficientnet_b0')
        
    Returns:
        Loaded model in eval mode
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_map = {
        'resnet18': lambda: torchvision.models.resnet18(weights='IMAGENET1K_V1'),
        'resnet34': lambda: torchvision.models.resnet34(weights='IMAGENET1K_V1'),
        'resnet50': lambda: torchvision.models.resnet50(weights='IMAGENET1K_V2'),
        'resnet101': lambda: torchvision.models.resnet101(weights='IMAGENET1K_V2'),
        'resnet152': lambda: torchvision.models.resnet152(weights='IMAGENET1K_V2'),
        'efficientnet_b0': lambda: torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1'),
        'efficientnet_b1': lambda: torchvision.models.efficientnet_b1(weights='IMAGENET1K_V1'),
        'efficientnet_b2': lambda: torchvision.models.efficientnet_b2(weights='IMAGENET1K_V1'),
        'efficientnet_b3': lambda: torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1'),
        'vit_b_16': lambda: torchvision.models.vit_b_16(weights='IMAGENET1K_V1'),
        'vit_b_32': lambda: torchvision.models.vit_b_32(weights='IMAGENET1K_V1'),
        'densenet121': lambda: torchvision.models.densenet121(weights='IMAGENET1K_V1'),
        'densenet169': lambda: torchvision.models.densenet169(weights='IMAGENET1K_V1'),
        'mobilenet_v2': lambda: torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1'),
        'mobilenet_v3_large': lambda: torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V1'),
        'mobilenet_v3_small': lambda: torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1'),
    }
    
    if model_name not in model_map:
        available_models = list(model_map.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available_models}")
    
    print(f"Loading model: {model_name}")
    model = model_map[model_name]()
    model = model.to(device)
    model.eval()
    
    return model


def get_available_models() -> List[str]:
    """Get list of available model names."""
    return [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        'vit_b_16', 'vit_b_32',
        'densenet121', 'densenet169',
        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small'
    ]