# Optimized Conformal Classification

## Quick Start

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/TaoWangPenn/Optimized-Conformal-Classification.git
cd Optimized-Conformal-Classification

# Install dependencies
pip install -r requirements.txt
```

### Simple Example
```python
from src.conformalizer import ConformalModel
from src.utils import validate

# Create conformal model with M-criterion optimization
conformal_model = ConformalModel(
    model=your_model,
    calib_loader=calib_loader,
    alpha=0.05,  # 95% coverage guarantee
    method='m-criterion-simple'  # Advanced optimization method
)

# Validate performance with coverage guarantee
results = validate(val_loader, conformal_model)
top1, top5, coverage, avg_size, prob_gt1 = results
print(f"Coverage: {coverage:.3f} (Target: {1-0.05:.3f})")
print(f"Average Set Size: {avg_size:.3f}")
```

### Run Complete Examples
```bash
# Model-based approach (end-to-end)
python example_model_based.py /path/to/imagenet/validation --alpha 0.05 --num_calib 20000

# Logits-based approach (high performance)
python example_logits_based.py --logits_file models/logits/resnet152_logits.pkl --alpha 0.05

# Exploratory data analysis on conformal methods
python eda.py --data /path/to/imagenet/validation --alpha 0.05
```

## Usage Guide

This library provides **two different approaches** for conformal prediction:

### 1. **ConformalModel** - High-level End-to-End Interface
Perfect for quick experiments and prototyping:

```python
from src.conformalizer import ConformalModel
from src.utils import validate
import torchvision

# Load your model and data
model = torchvision.models.resnet152(weights='IMAGENET1K_V2')
# calib_loader, val_loader = your_data_loaders

# Create conformal model (handles everything automatically)
conformal_model = ConformalModel(
    model=model,
    calib_loader=calib_loader,
    alpha=0.05,  # 95% coverage guarantee
    method='m-criterion-simple',  # Advanced M-criterion optimization
    verbose=True
)

# Direct inference with prediction sets
logits, prediction_sets = conformal_model(test_inputs)

# Comprehensive validation
results = validate(val_loader, conformal_model)
top1, top5, coverage, avg_size, prob_gt1 = results
```

### 2. **ConformalPredictor** - Low-level Logits-based Interface
Optimized for performance and large-scale experiments with multiple methods and models:

```python
from src.conformalizer import ConformalPredictor
from src.utils import validate, get_logits_targets, platt_scale

# Step 1: One-time calibration setup with Platt scaling
T, calib_logits = platt_scale(model, calib_loader)

# Step 2: Create efficient predictor with advanced methods
conformal_predictor = ConformalPredictor(
    calib_logits=calib_logits,  # TensorDataset directly
    alpha=0.05,
    method='m-criterion-simple',  # Or 'm-criterion-weighted', 'length-optimization'
    temperature=T.item(),
    verbose=True
)

# Step 3: Pre-compute validation logits (efficient!)
val_logits = get_logits_targets(model, val_loader)

# Step 4: Fast validation with no redundant forward passes
results = validate(val_loader, conformal_predictor, val_logits=val_logits)
top1, top5, coverage, avg_size, prob_gt1 = results

# Direct prediction on new logits
test_logits = model(test_inputs)
prediction_sets = conformal_predictor.predict(test_logits.cpu())
```

#### **Pre-computed Logits** 
To use ConformalPredictor class:

```python
# Pre-compute logits once (saves significant time for multiple experiments)
python models/precompute_logits.py --model resnet152 --data_path /path/to/imagenet/val

# Then use the efficient logits-based example
python example_logits_based.py --logits_file models/logits/resnet152_logits.pkl

# Available models: resnet18/34/50/101/152, efficientnet_b0-3, vit_b_16/32, densenet121/169, mobilenet_v2/v3
python models/precompute_logits.py --list_models
```

## 🎯 Available Methods

| Method | Description | Deterministic | Optimization | Performance |
|--------|-------------|---------------|--------------|-------------|
| `'naive'` | Simple cumulative threshold conformal | ✅ | None | Fastest |
| `'length-optimization'` | Length-optimized prediction sets | ✅ | Convex | Fast |
| `'m-criterion-simple'` | M-criterion with theoretical guarantees | ✅ | Non-convex | Medium |
| `'m-criterion-weighted'` | Weighted M-criterion variant | ✅ | Non-convex | Medium |
| `'m-criterion-randomized'` | Randomized M-criterion method | ❌ | Non-convex | Medium |
| `'length-optimization-randomized'` | Randomized length optimization | ❌ | Convex | Fast |

### Advanced M-criterion Features:
- **SLSQP Solver**: Sequential Least Squares Programming for constrained optimization
- **PGA Solver**: Projected Gradient Ascent with Armijo line search
- **Multi-start Optimization**: Robust optimization with multiple initializations
- **Gamma Conjecture**: Theoretical initialization for faster convergence
- **Dimensionality Reduction**: Automatic problem size reduction based on probability thresholds

```python
# Get available methods and detailed information
from src.conformalizer import ConformalModel
print(ConformalModel.get_available_methods())

# Access advanced M-criterion optimization directly
from src.optimization.M_criterion import OptimizationProblem, SLSQPSolver, PGASolver
```

## 📖 API Reference

### ConformalModel (High-level Interface)
```python
ConformalModel(
    model,                    # PyTorch model
    calib_loader,            # Calibration data loader
    alpha,                   # Miscoverage rate (e.g., 0.05 for 95% coverage)
    method='m-criterion-simple',  # Conformal method with optimization
    T=None,                  # Temperature (auto-computed with Platt scaling if None)
    calib_logits=None,       # Pre-computed logits (optional, for efficiency)
    verbose=True             # Print progress and optimization details
)
```

### ConformalPredictor (Low-level Interface)
```python
ConformalPredictor(
    calib_logits,            # TensorDataset from get_logits_targets()
    alpha,                   # Miscoverage rate (e.g., 0.05 for 95% coverage)
    method='m-criterion-simple',  # Advanced conformal method
    temperature=1.0,         # Temperature scaling parameter from Platt scaling
    verbose=True             # Print calibration and optimization progress
)
```

### Validation Function
```python
validate(
    val_loader,              # Validation data loader
    model_or_predictor,      # ConformalModel or ConformalPredictor
    base_model=None,         # For ConformalPredictor (if no val_logits)
    val_logits=None,         # Pre-computed logits (recommended for efficiency)
    print_bool=True          # Print real-time progress
)
# Returns: (top1_acc, top5_acc, coverage, avg_set_size, prob_size_gt1)
```

### Core Utility Functions
```python
# Pre-compute logits for maximum efficiency
val_logits = get_logits_targets(model, val_loader)

# Temperature scaling (Platt scaling) for probability calibration
T, calib_logits = platt_scale(model, calib_loader)

# Load pre-computed logits (for research workflows)
from src.utils import load_logits, split_logits
logits_dataset = load_logits('resnet152', './models/logits')
calib_logits, val_logits = split_logits(logits_dataset, num_calib=20000, num_val=10000)
```

### Advanced M-criterion Optimization
```python
from src.optimization.M_criterion import OptimizationProblem, SLSQPSolver, PGASolver

# Create optimization problem for a probability vector
problem = OptimizationProblem(p_hat, alpha=0.05, threshold=None)

# Solve with different algorithms
slsqp_solver = SLSQPSolver(problem, tol=1e-9, max_iter=1000)
pga_solver = PGASolver(problem, tol=1e-9, max_iter=5000)

# Multi-start optimization for robustness
from src.optimization.M_criterion import find_best_solution_with_multistart
best_result = find_best_solution_with_multistart(problem, slsqp_solver, num_starts=10)
```

## Architecture Overview

```
┌─────────────┐    ┌───────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Input Images│ -> │   Model   │ -> │     Logits       │ -> │ConformalPredicto│
│             │    │ (ResNet,  │    │ (Pre-computed/   │    │r(M-criterion    │ 
│             │    │  ViT)     │    │  Cached)         │    │  Optimization)  │
│             │    │           │    |                  │    |                 │
└─────────────┘    └───────────┘    └──────────────────┘    └─────────────────┘
                                               ↑                        ↓
                                     ┌─────────────────┐    ┌─────────────────┐
                                     │ Temperature     │    │ Prediction Sets │
                                     │ Scaling (Platt) │    │ with Coverage   │
                                     └─────────────────┘    │ Guarantees      │
                                                            └─────────────────┘
```

##  Problem Background
##  Theoretical Background

### Conformal Prediction Framework
Given data $(x, y) \in \mathcal{X} \times \mathcal{Y}$ where $\mathcal{Y} = [K]$ for $K$-class classification.

**Randomized prediction set** $\Gamma$:
$$\forall x \in \mathcal{X}, \quad \Gamma(x) \in [0,1]^{K}$$

where $[\Gamma(x)]_y = \mathbb{P}(y \in C(x))$ represents the inclusion probability for class $y$.

**Coverage guarantee**:
$$\mathbb{P}(Y \in C(X)) = \mathbb{E}_{X} \left[ \langle \Gamma(X), p(\cdot|X) \rangle \right] \geq 1-\alpha$$

### M-criterion Optimization Problem

The M-criterion method solves the constrained optimization problem:
$$\begin{align}
&\min_{\gamma} \mathbb{P} \left[ |C(X)| > 1 \right]\\
&\text{subject to } \sum_{i=1}^K p_i\gamma_i \geq 1-\alpha, \quad 0 \leq \gamma_i \leq 1
\end{align}$$

**Objective function** (probability of non-singleton sets):
$$g(\gamma) = \prod_{i=1}^K (1 - \gamma_i) + \sum_{i=1}^K \gamma_i \prod_{j \neq i} (1-\gamma_j)$$

**Log-domain optimization** (numerical stability):
$$\log g(\gamma) = \text{LSE}\left( \sum_{i=1}^K \log(1-\gamma_i), \{\log(\gamma_k) + \sum_{j \neq k} \log(1-\gamma_j)\}_{k=1}^K \right)$$

### Solver Implementations

#### SLSQP (Sequential Least Squares Programming)
- **Method**: Quasi-Newton with BFGS approximation
- **Constraints**: Equality and inequality constraints handled via augmented Lagrangian

#### PGA (Projected Gradient Ascent)  
- **Method**: First-order with Armijo line search
- **Projection**: Efficient Brent's method for simplex constraints

#### Gamma Conjecture 
**Theoretical result**: For sorted probabilities $p_1 \geq p_2 \geq \ldots \geq p_K$: when $(1-\alpha) /2 < p_1 < 1-\alpha$, a strict local maximizer:
$$\gamma^*_1 = 1, \quad \gamma^*_i = 1 - \frac{\sum_{j=1}^{m-1} p_j - (1-\alpha)}{(m-1) \cdot p_i} \text{ for } i = 2, \ldots, m$$
where $m$ is defined by
$$
p_m > \frac{\sum_{i=1}^m p_i-(1-\alpha)}{m-1} \geq p_{m+1}
$$
where by convention, $p_{K+1} \equiv 0$.

### Conformalization Methods

The framework implements seven distinct conformal prediction methods, each with different characteristics and optimization strategies:

#### 1. **Naive (Cumulative)** 
- **Method**: `naive`
- **Calibration**: Not required
- **Algorithm**: Selects classes in descending order of probability until cumulative probability reaches $1-\alpha$
- **Mathematical Formulation**: 
  $$C(x) = \{i_1, i_2, \ldots, i_m\} \text{ where } \sum_{j=1}^m p_{i_j} \geq 1-\alpha$$

#### 2. **Length Optimization**
- **Method**: `length-optimization`
- **Calibration**: Required (conformity score threshold)
- **Algorithm**: Minimizes prediction set size while maintaining coverage guarantee
- **Mathematical Formulation**:
  $$C(x) = \{i : \gamma^{LO} \geq \tau\} \text{ where } \gamma^{LO} \text{ solves the Length optimization}$$
- **Conformity Score**: $\gamma_i$ (direct inclusion probability)

#### 3. **Length Optimization Weighted**
- **Method**: `length-optimization-weighted`
- **Calibration**: Required (conformity score threshold)
- **Algorithm**: Weighted version of length optimization using $p_i \cdot \gamma_i$ as conformity score
- **Mathematical Formulation**:
  $$C(x) = \{i : p_i \cdot \gamma^{LO}_i > \tau\} \text{ where } \gamma^{LO} \text{ solves the length optimization}$$
- **Conformity Score**: $p_i \cdot \gamma_i$ (weighted inclusion probability)

#### 4. **M-Criterion Simple**
- **Method**: `m-criterion-simple`
- **Calibration**: Required (conformity score threshold)
- **Algorithm**: Uses M-criterion optimization to minimize probability of non-singleton sets
- **Mathematical Formulation**:
  $$C(x) = \{i : \gamma^*_i \geq \tau\} \text{ where } \gamma^* \text{ solves the M-criterion optimization}$$
- **Conformity Score**: $\gamma_i$ (direct inclusion probability)

#### 5. **M-Criterion Weighted**
- **Method**: `m-criterion-weighted`
- **Calibration**: Required (conformity score threshold)
- **Algorithm**: Weighted version of M-criterion using $p_i \cdot \gamma_i$ as conformity score
- **Mathematical Formulation**:
  $$C(x) = \{i : p_i \cdot \gamma^*_i > \tau\} \text{ where } \gamma^* \text{ solves the M-criterion optimization}$$
- **Conformity Score**: $p_i \cdot \gamma_i$ (weighted inclusion probability)

#### 6. **M-Criterion Randomized**
- **Method**: `m-criterion-randomized`
- **Calibration**: Not required (direct sampling)
- **Algorithm**: Uses M-criterion $\gamma$ values as sampling probabilities
- **Mathematical Formulation**:
  $$C(x) = \{i : \text{Bernoulli}(\gamma^*_i) = 1\} \text{ where } \gamma^* \text{ solves the M-criterion optimization}$$

#### 7. **Length Optimization Randomized**
- **Method**: `length-optimization-randomized`
- **Calibration**: Not required (direct sampling)
- **Algorithm**: Uses length optimization $\gamma$ values as sampling probabilities
- **Mathematical Formulation**:
  $$C(x) = \{i : \text{Bernoulli}(\gamma^{LO}_i) = 1\} \text{ where } \gamma^{LO} \text{ solves the length optimization}$$
  

#### Implementation Details

All methods are implemented through a factory pattern (`MethodFactory`) that provides:
- Unified interface for all conformal methods
- Automatic method validation and instantiation
- Consistent API for conformity score computation and prediction set generation
- Support for both deterministic and randomized approaches

## 📊 Experimental Results

### Model: Resnet-152; Dataset: ImageNet Validation
### EDA:

**Dataset Statistics:**
- Total validation samples: 50,000
- Number of classes: 1,000
- Threshold for Hard Case: 0.4750
- Samples with $\max \hat{p}_i$ ≤ threshold: 3,125 (6.25%)

**Confidence Percentiles $\max \hat{p}_i$:**
- 1st percentile: 0.2465
- 5th percentile: 0.4388
- 10th percentile: 0.5553
- 25th percentile: 0.8263
- 50th percentile: 0.9558
- 75th percentile: 0.9738
- 90th percentile: 0.9816

### Resnet-152; ImageNet Validation (Average across 100 data splits, α=0.05)

| Method | Top-1 Acc | Top-5 Acc | Coverage | Avg Size | P(Size>1) | Time (s) |
|--------|-----------|-----------|----------|----------|-----------|----------|
| Naive | 82.36±0.24% | 95.92±0.11% | 96.94±0.29% | 46.72±6.48 | 46.84±2.34% | 0.92±0.09 |
| Length-Opt | 82.36±0.24% | 95.92±0.11% | 94.99±0.23% | 46.32±6.43 | 36.71±1.51% | 1.57±0.01 |
| Length-Opt Weighted | 82.36±0.24% | 95.92±0.11% | 95.01±0.21% | 2.56±0.09 | 41.61±0.88% | 1.63±0.15 |
| M-Criterion Simple | 82.36±0.24% | 95.92±0.11% | 95.00±0.22% | 9.44±1.73 | **36.05±1.03%** | 3.70±0.40 |
| M-Criterion Weighted | 82.36±0.24% | 95.92±0.11% | 95.01±0.21% | **2.52±0.08** | 41.19±0.90% | 3.71±0.40 |
| M-Criterion Randomized | 82.36±0.24% | 95.92±0.11% | 95.02±0.45% | 57.94±7.85 | 42.21±2.40% | 2.22±0.20 |
| Length-Opt Randomized | 82.36±0.24% | 95.92±0.11% | 95.04±0.49% | 46.50±6.47 | 42.43±2.41% | 1.15±0.02 |

*Results averaged over 100 different calibration/validation splits*  
*Bold values indicate best performance in average set size*

### Bootstrap 95% Confidence Intervals (1000 samples)

| Method | Coverage CI | Avg Size CI |
|--------|-------------|-------------|
| Naive | [96.88%, 96.99%] | [45.44, 47.98] |
| Length-Opt | [94.95%, 95.04%] | [45.08, 47.49] |
| Length-Opt Weighted | [94.97%, 95.05%] | [2.55, 2.58] |
| M-Criterion Simple | [94.96%, 95.05%] | [9.10, 9.80] |
| M-Criterion Weighted | [94.97%, 95.05%] | [2.50, 2.53] |
| M-Criterion Randomized | [94.93%, 95.11%] | [56.42, 59.37] |
| Length-Opt Randomized | [94.95%, 95.13%] | [45.30, 47.79] |
