# src/utils.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import time
import pickle
import random
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ============================================================================
# Reproducibility
# ============================================================================

def set_random_seeds(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Performance Monitoring
# ============================================================================

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# ============================================================================
# Model Validation and Metrics
# ============================================================================

def validate(val_loader, model_or_predictor, base_model=None, val_logits=None, print_bool=True):
    """
    Universal validation function supporting multiple input types.
    
    Args:
        val_loader: Validation data loader (for ConformalModel or when base_model provided)
        model_or_predictor: Either:
            - ConformalModel instance (high-level wrapper)
            - ConformalPredictor instance (requires base_model or val_logits)
        base_model: Required when using ConformalPredictor without pre-computed logits
        val_logits: Pre-computed validation logits (TensorDataset) for ConformalPredictor
        print_bool: Whether to print progress
    
    Returns:
        Tuple: (top1_accuracy, top5_accuracy, coverage, avg_set_size, prob_size_gt1)
    """
    
    if hasattr(model_or_predictor, 'forward') and hasattr(model_or_predictor, 'conformal_predictor'):
        model_or_predictor.eval()
        data_loader = val_loader
        
        def get_predictions(x, target):
            if torch.cuda.is_available():
                target = target.cuda()
                x = x.cuda()
            logits, prediction_sets = model_or_predictor(x)
            return logits, prediction_sets, target, x.shape[0]
            
    elif hasattr(model_or_predictor, 'predict'):
        if val_logits is not None:
            data_loader = tdata.DataLoader(val_logits, batch_size=val_loader.batch_size, shuffle=False, num_workers=0)
            
            def get_predictions(logits, target):
                if torch.cuda.is_available():
                    target = target.cuda()
                    if target.is_cuda:
                        logits = logits.cuda()
                prediction_sets = model_or_predictor.predict(logits.cpu())
                return logits, prediction_sets, target, logits.shape[0]
                
        elif base_model is not None:
            base_model.eval()
            data_loader = val_loader
            
            def get_predictions(x, target):
                if torch.cuda.is_available():
                    target = target.cuda()
                    x = x.cuda()
                logits = base_model(x)
                prediction_sets = model_or_predictor.predict(logits.detach().cpu())
                return logits, prediction_sets, target, x.shape[0]
        else:
            raise ValueError(
                "When using ConformalPredictor, provide either:\n"
                "- base_model (for on-the-fly logit computation), or\n"
                "- val_logits (pre-computed logits, recommended for efficiency)"
            )
    else:
        raise ValueError(
            "Invalid input combination. Please provide:\n"
            "- ConformalModel instance, or\n" 
            "- ConformalPredictor instance + (base_model or val_logits)"
        )
    
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('Coverage')
        size = AverageMeter('Size')
        prob_gt1 = AverageMeter('Size>1 Prob') 
        
        end = time.time()
        N = 0
        
        for i, (x, target) in enumerate(data_loader):
            logits, prediction_sets, target, batch_size = get_predictions(x, target)
            
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            cvg, sz = coverage_size(prediction_sets, target.cpu() if target.is_cuda else target)
            p_gt1 = size_gt1_prob(prediction_sets) 

            top1.update(prec1.item() / 100.0, n=batch_size)
            top5.update(prec5.item() / 100.0, n=batch_size)
            coverage.update(cvg, n=batch_size)
            size.update(sz, n=batch_size)
            prob_gt1.update(p_gt1, n=batch_size) 

            batch_time.update(time.time() - end)
            end = time.time()
            N += batch_size
            
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg: {coverage.val:.3f} ({coverage.avg:.3f}) | Size: {size.val:.3f} ({size.avg:.3f}) | Size>1: {prob_gt1.val:.3f} ({prob_gt1.avg:.3f})', end='')
    
    if print_bool:
        print('') 

    return top1.avg, top5.avg, coverage.avg, size.avg, prob_gt1.avg


def validate_conformal_predictor(val_loader, base_model, conformal_predictor, print_bool=True):
    """
    Backward compatibility alias. Recommend using validate() function instead.
    """
    return validate(val_loader, conformal_predictor, base_model, print_bool)


def coverage_size(S, targets):
    covered = 0
    total_size = 0
    for i in range(len(S)):
        if targets[i].item() in S[i]:
            covered += 1
        total_size += len(S[i])
    
    return covered / len(S), total_size / len(S)

def size_gt1_prob(S):
    count_gt1 = 0
    if len(S) == 0:
        return 0
        
    for s_i in S:
        if len(s_i) > 1:
            count_gt1 += 1
            
    return count_gt1 / len(S)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# ============================================================================
# Data Processing and Logits Extraction
# ============================================================================

def get_logits_targets(model, loader):
    num_samples = len(loader.dataset)
    
    if hasattr(loader.dataset, 'dataset') and hasattr(loader.dataset.dataset, 'classes'):
        num_classes = len(loader.dataset.dataset.classes)
    elif hasattr(loader.dataset, 'classes'):
        num_classes = len(loader.dataset.classes)
    else:
        with torch.no_grad():
            sample_batch = next(iter(loader))
            sample_input = sample_batch[0][:1]  
            if torch.cuda.is_available():
                sample_input = sample_input.cuda()
            sample_output = model(sample_input)
            num_classes = sample_output.shape[1]

    all_logits = torch.empty((num_samples, num_classes), dtype=torch.float32, device='cpu')
    all_labels = torch.empty(num_samples, dtype=torch.int64, device='cpu')

    print('Computing logits for model (only happens once).')
    current_pos = 0
    with torch.no_grad():
        for x, targets in tqdm(loader):
            if torch.cuda.is_available():
                x = x.cuda()

            batch_logits = model(x).detach().cpu()
            batch_size = x.shape[0]

            all_logits[current_pos:current_pos+batch_size, :] = batch_logits
            all_labels[current_pos:current_pos+batch_size] = targets.cpu()
            current_pos += batch_size

    return torch.utils.data.TensorDataset(all_logits, all_labels)

# ============================================================================
# Temperature Scaling (Platt Scaling)
# ============================================================================

def platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    """
    Tunes a temperature parameter T for Platt scaling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nll_criterion = nn.CrossEntropyLoss().to(device)

    T = nn.Parameter(torch.tensor([1.3], device=device))

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            x = x.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            loss = nll_criterion(x / T, targets)
            loss.backward()
            optimizer.step()
            
        if abs(T_old - T.item()) < epsilon:
            break
            
    return T


def platt_scale(model, calib_loader, verbose=True):
    """
    Perform Platt scaling on a model using calibration data.
    """
    if verbose:
        print("Begin Platt scaling.")
    
    logits_dataset = get_logits_targets(model, calib_loader)
    logits_loader = torch.utils.data.DataLoader(
        logits_dataset, 
        batch_size=calib_loader.batch_size, 
        shuffle=False, 
        pin_memory=True
    )

    T = platt_logits(logits_loader)

    if verbose:
        print(f"Optimal T={T.item()}")
    
    return T, logits_dataset


# ============================================================================
# Logits Loading and Splitting Functions
# ============================================================================

def load_logits(model_name: str, logits_dir: str = './logits_cache') -> tdata.TensorDataset:
    """
    Load pre-computed logits for a model.
    """
    logits_file = Path(logits_dir) / f"{model_name}_logits.pkl"
    
    if not logits_file.exists():
        raise FileNotFoundError(f"Logits file not found: {logits_file}")
    
    with open(logits_file, 'rb') as f:
        logits_dataset = pickle.load(f)
    
    print(f"Loaded logits for {model_name}: {logits_dataset.tensors[0].shape}")
    return logits_dataset


def split_logits(
    logits_dataset: tdata.TensorDataset, 
    num_calib: Optional[int] = None,
    num_val: Optional[int] = None,
    calib_ratio: float = 0.3
) -> Tuple[tdata.TensorDataset, tdata.TensorDataset]:
    """
    Split logits dataset into calibration and validation sets.
    """
    total_size = len(logits_dataset)
    
    if num_calib is not None:
        calib_size = min(num_calib, total_size)
    else:
        calib_size = int(calib_ratio * total_size)
    
    if num_val is not None:
        val_size = min(num_val, total_size - calib_size)
    else:
        val_size = total_size - calib_size
    
    if calib_size + val_size > total_size:
        if num_calib is not None and num_val is not None:
            val_size = total_size - calib_size
            if val_size <= 0:
                raise ValueError(f"Not enough data: need {calib_size} calib + {num_val} val, but only have {total_size} total")
    
    indices = torch.randperm(total_size)
    calib_indices = indices[:calib_size]
    val_indices = indices[calib_size:calib_size + val_size]
    
    all_logits = logits_dataset.tensors[0]
    all_targets = logits_dataset.tensors[1]
    
    calib_logits = all_logits[calib_indices]
    calib_targets = all_targets[calib_indices]
    
    val_logits = all_logits[val_indices]
    val_targets = all_targets[val_indices]
    
    calib_dataset = tdata.TensorDataset(calib_logits, calib_targets)
    val_dataset = tdata.TensorDataset(val_logits, val_targets)
    
    print(f"Split dataset: {len(calib_dataset)} calibration, {len(val_dataset)} validation (from {total_size} total)")
    return calib_dataset, val_dataset


def load_and_split_logits(
    model_name: str, 
    logits_dir: str = './logits_cache',
    num_calib: Optional[int] = None,
    num_val: Optional[int] = None,
    calib_ratio: float = 0.3
) -> Tuple[tdata.TensorDataset, tdata.TensorDataset]:
    """
    Load logits and split into calibration and validation sets.
    """
    logits_dataset = load_logits(model_name, logits_dir)
    return split_logits(logits_dataset, num_calib, num_val, calib_ratio)
