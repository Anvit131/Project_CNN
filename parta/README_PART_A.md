# Project_CNN

# CNN Classifier with W&B Hyperparameter Sweeps

This project builds a **custom Convolutional Neural Network (CNN)** to classify images from the [iNaturalist 12K Dataset](https://www.kaggle.com/datasets/jhmontalvo01/inaturalist-2021-mini). It uses **PyTorch** for modeling and **Weights & Biases (wandb)** for experiment tracking and hyperparameter optimization through **Bayesian Sweeps**.

---

##  Dataset Setup

- The dataset is expected at:  
  `/kaggle/input/dataset/inaturalist_12K/train`

- Data is loaded using `torchvision.datasets.ImageFolder`.

### Data Preparation and Augmentation

```python
data_augmentation = random.choice([True, False])
transform = transforms.Compose([...])
```

- If `data_augmentation` is `True`, the images go through:
  - Random cropping
  - Horizontal flip
  - Color jitter
  - Random rotation

- Otherwise, a simple resize and normalization is applied.

###  Splitting and Loading

```python
train_dataset, val_dataset = random_split(full_train_dataset, [80%, 20%])
```

- Training and validation datasets are created with `DataLoader`.

---

##  Custom Convolutional Neural Network (CNN)

A flexible CNN with exactly **5 convolutional layers** and support for:

- Custom filter sizes
- Activation functions
- Batch normalization
- Dropout
- Dynamically computed flattened layer

###  Model Class

```python
class CNN_Model(nn.Module):
    def __init__(filter_sizes, dense_neurons, activation, ...)
```

####  Key Features

| Component        | Description |
|------------------|-------------|
| `filter_sizes`   | List of filter sizes for conv layers |
| `activation`     | Supports `"relu"`, `"gelu"`, `"silu"`, `"mish"` |
| `dropout`        | Optional dropout after each layer |
| `batch_norm`     | Optional BatchNorm2d |
| `flatten_dim`    | Computed dynamically via dummy input |
| `output`         | Output layer has 10 neurons (for 10 classes) |

---

## Model Training

The training is handled by the `train_cnn_model()` function.

### Training Loop

- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Tracks:
  - Training/Validation loss
  - Accuracy
  - Logs to wandb

### Model Checkpoint

```python
torch.save(model.state_dict(), "best_model.pth")
```

- Saves model only when validation accuracy improves.

---

## Hyperparameter Optimization with W&B Sweeps

```python
sweep_config = {
    "method": "bayes",
    ...
}
```

### Sweep Goal

- Maximize **`val_accuracy`**

### Parameters Tuned

| Parameter        | Values |
|------------------|--------|
| `epochs`         | `[5, 10, 15, 20]` |
| `filter_sizes`   | `[16, 32, 32, 32, 32]`, `[128, 128, 256, 256, 512]`, `[32, 64, 64, 128, 128]`, `[32, 64, 128, 128, 256]` |
| `dense_neurons`  | `[128, 256, 512]` |
| `activation`     | `relu`, `silu`, `gelu`, `mish` |
| `learning_rate`  | `[1e-3, 1e-4]` |
| `dropout`        | `[0.2, 0.3]` |
| `batch_norm`     | `[True, False]` |
| `weight_decay`   | `[0, 0.0005]` |
| `data_augmentation` | `[True, False]` |

---

## Running the Sweep

Define the sweep logic:

```python
def train_wandb():
    with wandb.init(project="Assignment_2_CNN"):
        ...
```

### Launch the sweep:

```python
sweep_id = wandb.sweep(sweep_config, project="Assignment_2_CNN")
wandb.agent(sweep_id, function=train_wandb, count=15)
```

- This will run **15 trials** with different hyperparameter combinations, tracking everything on your W&B dashboard.

---

## Prerequisites

- `wandb` must be installed and logged in:

```bash
pip install wandb
wandb login
```

---


The codebase can be modularized into the following structure for better maintainability:

project/

├── model.py              # Model definition and customization

├── train.py              # Main training script

├── get_dataloaders.py               # Dataset loading and transforms


├── config.py             # Configuration variables

├── train_wandb           # Wandb training

