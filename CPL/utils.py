import torch
import numpy as np
import random
from torch.utils.data import TensorDataset

def set_random_seeds(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def split_reps_logits(reps_dataset, logits_dataset, num_tune, num_calib, num_val):
    """
    Splits representations and logits datasets into tune, calibration, and validation sets.
    """
    total_samples = len(reps_dataset)
    if total_samples != len(logits_dataset):
        raise ValueError("Representation and logit datasets must have the same number of samples.")
    
    if num_tune + num_calib + num_val > total_samples:
        raise ValueError("Sum of tune, calibration, and validation samples exceeds total.")

    indices = torch.randperm(total_samples).tolist()
    
    tune_indices = indices[:num_tune]
    calib_indices = indices[num_tune:num_tune + num_calib]
    val_indices = indices[num_tune + num_calib:num_tune + num_calib + num_val]

    def _split_dataset(dataset, indices):
        return dataset.tensors[0][indices], dataset.tensors[1][indices]

    tune_reps, tune_labels_reps = _split_dataset(reps_dataset, tune_indices)
    calib_reps, calib_labels_reps = _split_dataset(reps_dataset, calib_indices)
    val_reps, val_labels_reps = _split_dataset(reps_dataset, val_indices)

    tune_logits, tune_labels_logits = _split_dataset(logits_dataset, tune_indices)
    calib_logits, calib_labels_logits = _split_dataset(logits_dataset, calib_indices)
    val_logits, val_labels_logits = _split_dataset(logits_dataset, val_indices)

    assert torch.equal(tune_labels_reps, tune_labels_logits)
    assert torch.equal(calib_labels_reps, calib_labels_logits)
    assert torch.equal(val_labels_reps, val_labels_logits)

    tune_data = (tune_reps.numpy(), tune_logits, tune_labels_logits)
    calib_data = (calib_reps.numpy(), calib_logits, calib_labels_logits)
    val_data = (val_reps.numpy(), val_logits, val_labels_logits)
    
    return tune_data, calib_data, val_data

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

def validate_cpl(prediction_sets, logits, labels):
    """
    Validates the performance of CPL-generated prediction sets, calculating detailed metrics.

    Args:
        prediction_sets (list[np.ndarray]): A list of prediction sets.
        logits (torch.Tensor): The output logits for the validation set.
        labels (torch.Tensor): A tensor of true labels.

    Returns:
        dict: A dictionary containing the metrics:
            - 'Top-1 Accuracy', 'Top-5 Accuracy'
            - 'Coverage': The empirical coverage rate.
            - 'Average Set Size': The average size of the prediction sets.
            - 'Prob(Size > 1)': The proportion of sets with size greater than 1.
            - 'Prob(Empty)': The proportion of empty sets.
    """
    num_samples = len(labels)
    if num_samples == 0:
        return {'Top-1 Accuracy': 0, 'Top-5 Accuracy': 0, 'Coverage': 0, 'Average Set Size': 0, 'Prob(Size > 1)': 0, 'Prob(Empty)': 0}

    num_classes = logits.size(1)
    topk = (1, min(5, num_classes))
    prec1, prec5 = accuracy(logits, labels, topk=topk)
    top1_acc = prec1.item() / 100.0
    top5_acc = prec5.item() / 100.0

    labels_np = labels.numpy()
    covered = 0
    total_size = 0
    count_gt1 = 0
    count_empty = 0

    for i in range(num_samples):
        pred_set = prediction_sets[i]
        true_label = labels_np[i]
        set_size = len(pred_set)

        if true_label in pred_set:
            covered += 1
        
        if set_size > 1:
            count_gt1 += 1
        
        if set_size == 0:
            count_empty += 1
            
        total_size += set_size

    metrics = {
        'Top-1 Accuracy': top1_acc,
        'Top-5 Accuracy': top5_acc,
        'Coverage': covered / num_samples,
        'Average Set Size': total_size / num_samples,
        'Prob(Size > 1)': count_gt1 / num_samples,
        'Prob(Empty)': count_empty / num_samples
    }
    
    return metrics

def split_for_fullbatch(calib_reps, calib_scores_true, calib_scores_all, split_ratio=0.5):
    n = calib_reps.shape[0]
    n_theta = int(n * split_ratio)
    indices = torch.randperm(n)
    idx_theta = indices[:n_theta]
    idx_c = indices[n_theta:]

    theta_reps = calib_reps[idx_theta]
    theta_scores_true = calib_scores_true[idx_theta]
    theta_scores_all = calib_scores_all[idx_theta]

    c_reps = calib_reps[idx_c]
    c_scores_true = calib_scores_true[idx_c]
    c_scores_all = calib_scores_all[idx_c]
    return (theta_reps, theta_scores_true, theta_scores_all,
            c_reps, c_scores_true, c_scores_all)
