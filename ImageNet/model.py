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
    print(f"Using device: {device}")
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    model_map = {
        'resnet152v1': lambda: torchvision.models.resnet152(weights='IMAGENET1K_V1'),
        'resnet152': lambda: torchvision.models.resnet152(weights='IMAGENET1K_V2'),
        'efficientnet_v2_l': lambda: torchvision.models.efficientnet_v2_l(weights='IMAGENET1K_V1'),
        'convnext_base': lambda: torchvision.models.convnext_base(weights='IMAGENET1K_V1'),
        'swin_v2_b': lambda: torchvision.models.swin_v2_b(weights='IMAGENET1K_V1'),
        'vit_h_14': lambda: torchvision.models.vit_h_14(weights='IMAGENET1K_SWAG_E2E_V1'),
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
        'resnet152v1', 'resnet152', 'efficientnet_v2_l', 'convnext_base', 'swin_v2_b', 'vit_h_14'
    ]
