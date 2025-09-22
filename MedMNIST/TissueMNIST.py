import os
import PIL
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from pathlib import Path
import pickle
from medmnist import INFO, TissueMNIST

def precompute_reps_logits_and_labels(model, data_loader, device):
    all_logits = []
    all_labels = []
    hidden_reps_list = []
    
    model.eval()

    def hook(module, input, output):
        hidden_reps_list.append(input[0].detach().cpu())

    final_layer = model.fc
    handle = final_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc="Computing reps and logits for the test set"):
            outputs = model(inputs.to(device))
            all_logits.append(outputs.cpu())
            all_labels.append(targets.cpu().squeeze(-1))
            
    handle.remove()

    reps_tensor = torch.cat(hidden_reps_list, dim=0)
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    reps_dataset = data.TensorDataset(reps_tensor, labels_tensor)
    logits_dataset = data.TensorDataset(logits_tensor, labels_tensor)
    
    return reps_dataset, logits_dataset

def main():
    data_flag = 'tissuemnist'
    batch_size = 128
    image_size = 224
    data_root = '/path/to/your/medmnist'
    model_path = "/path/to/your/weights_tissuemnist/resnet50_224_3.pth"

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    print("==> Forcing load from 28x28 source file and resizing to 224x224 on the fly...")
    
    test_dataset = TissueMNIST(split='test', 
                               transform=data_transform, 
                               size=28, 
                               download=True,
                               root=data_root,
                               as_rgb=True)
    
    print(f"==> Test set loaded. Number of samples: {len(test_dataset)}")

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("==> Test set DataLoader created.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==> Using device: {device}")

    info = INFO[data_flag]
    n_classes = len(info['label'])
    
    model = models.resnet50(weights=None, num_classes=n_classes)
    
    try:
        state_dict = torch.load(model_path, map_location=device)['net']
        model.load_state_dict(state_dict)
        print(f"==> Pre-trained weights successfully loaded from: '{model_path}'!")
    except Exception as e:
        print(f"==> Error: Could not load weights.")
        print(f"==> Detailed error message: {e}")
        return

    model = model.to(device)
    model.eval()

    logits_output_dir = Path(__file__).parent / "logits"
    logits_output_dir.mkdir(parents=True, exist_ok=True)
    
    reps_output_dir = Path("/path/to/your/CPL_HiddenRep/medmnist")
    reps_output_dir.mkdir(parents=True, exist_ok=True)

    reps_dataset, logits_dataset = precompute_reps_logits_and_labels(model, test_loader, device)
    
    logits_output_path = logits_output_dir / f"{data_flag}_logits.pkl"
    with open(logits_output_path, 'wb') as f:
        pickle.dump(logits_dataset, f)
    print(f"==> Test set logits for TissueMNIST have been saved to: '{logits_output_path}'")

    reps_output_path = reps_output_dir / f"{data_flag}_hidden_reps.pkl"
    with open(reps_output_path, 'wb') as f:
        pickle.dump(reps_dataset, f)
    print(f"==> Test set hidden representations for TissueMNIST have been saved to: '{reps_output_path}'")
    
if __name__ == '__main__':
    main()
