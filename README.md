# Singleton-Optimized Conformal Prediction

This repository contains the implementation for Singleton-Optimized Conformal Prediction.

## Data Pre-processing and Logit Pre-computation

The `ImageNet`, `MedMNIST`, and `MMLU` directories contain the necessary scripts for data pre-processing and to pre-compute the logits for each dataset. These scripts must be run before applying the conformal prediction methods. Once the logits files are generated, you can proceed to run the different methods.

- **ImageNet**: The `ImageNet/precompute_logits.py` script loads various pre-trained models from `torchvision.models` (e.g., ResNet, EfficientNet, ViT) and computes logits on the ImageNet validation set. It handles model-specific image transformations automatically.

- **MedMNIST**: The `MedMNIST/TissueMNIST.py` script loads the respective dataset, resize the images to 224x224, and use a pre-trained ResNet-50 model to compute logits.

- **MMLU**: `MMLU/reprocess_mmlu.py` converts the raw MMLU dataset into the standard prompting format. `MMLU/precompute_logits.py` loads a large language model (e.g., Llama 3.1 8B-Instruct) to compute the logits for the four choices in each multiple-choice question.

## Running Conformal Prediction Methods

After obtaining the logits files, you can run the various conformal prediction methods using `main.py`.

You can run all methods at once with the following command. Make sure to replace `path/to/your/logits.pkl` with the actual path to your pre-computed logits file.

```bash
python main.py --methods naive raps pure-singleton las socop --randomized --use_numba --logits_file path/to/your/logits.pkl
```

Alternatively, you can run each method individually:

### Naive
```bash
python main.py --methods naive --logits_file path/to/your/logits.pkl
```

### RAPS (Regularized Adaptive Prediction Sets)
```bash
python main.py --methods raps --randomized --logits_file path/to/your/logits.pkl
```

### Pure Singleton
```bash
python main.py --methods pure-singleton --logits_file path/to/your/logits.pkl
```

### Least Ambiguous Sets (LAS)
```bash
python main.py --methods las --logits_file path/to/your/logits.pkl
```

### SOCOP (Our Method)
```bash
python main.py --methods socop --use_numba --logits_file path/to/your/logits.pkl
