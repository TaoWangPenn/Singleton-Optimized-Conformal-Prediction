import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

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

def validate(val_loader, model, print_bool=True):
    with torch.no_grad():
        batch_time = AverageMeter('batch_time')
        top1 = AverageMeter('top1')
        top5 = AverageMeter('top5')
        coverage = AverageMeter('M-criterion coverage')
        size = AverageMeter('M-criterion size')
        prob_gt1 = AverageMeter('Size>1 Prob') 
        
        model.eval()
        end = time.time()
        N = 0
        for i, (x, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                target = target.cuda()
                x = x.cuda()

            output, S = model(x)
            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            cvg, sz = coverage_size(S, target)
            p_gt1 = size_gt1_prob(S) 

            top1.update(prec1.item() / 100.0, n=x.shape[0])
            top5.update(prec5.item() / 100.0, n=x.shape[0])
            coverage.update(cvg, n=x.shape[0])
            size.update(sz, n=x.shape[0])
            prob_gt1.update(p_gt1, n=x.shape[0]) 

            batch_time.update(time.time() - end)
            end = time.time()
            N += x.shape[0]
            
            if print_bool:
                print(f'\rN: {N} | Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | Cvg@1: {top1.val:.3f} ({top1.avg:.3f}) | Cvg@5: {top5.val:.3f} ({top5.avg:.3f}) | Cvg@M: {coverage.val:.3f} ({coverage.avg:.3f}) | Size@M: {size.val:.3f} ({size.avg:.3f}) | Size>1: {prob_gt1.val:.3f} ({prob_gt1.avg:.3f})', end='')
    
    if print_bool:
        print('') 

    return top1.avg, top5.avg, coverage.avg, size.avg, prob_gt1.avg

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

def get_logits_targets(model, loader):
    num_samples = len(loader.dataset)
    num_classes = loader.dataset[0][0].shape[0] 

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

def platt_logits(calib_loader, max_iters=10, lr=0.01, epsilon=0.01):
    nll_criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        nll_criterion = nll_criterion.cuda()

    T = nn.Parameter(torch.Tensor([1.3]))
    if torch.cuda.is_available():
        T = T.cuda()

    optimizer = optim.SGD([T], lr=lr)
    for iter in range(max_iters):
        T_old = T.item()
        for x, targets in calib_loader:
            if torch.cuda.is_available():
                x = x.cuda()
                targets = targets.cuda()
            
            optimizer.zero_grad()
            loss = nll_criterion(x / T, targets)
            loss.backward()
            optimizer.step()
            
        if abs(T_old - T.item()) < epsilon:
            break
            
    return T
