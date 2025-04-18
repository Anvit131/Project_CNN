# Fine-Tuning ResNet50 on iNaturalist Dataset

This project demonstrates fine-tuning a pre-trained ResNet50 model on a 10-class subset of the iNaturalist dataset. It includes model modification, layer freezing strategies, training with logging via Weights & Biases (wandb), and visualization of metrics.


Dataset Structure

dataset_path/
    train/

dataset_path/
    val/
    
**Handling Different Image Dimensions**
using 

`from torchvision import transforms`

 Model Overview

Ie used the pretrained `torchvision.models.resnet50` Model:

- All layers are frozen initially.
- Last ResNet block (`layer4`) and final classifier (`fc`) are unfrozen.
- Final classification layer modified to predict 10 classes.

Sample code:

<model = models.resnet50(pretrained=True)>
# Strategy-
**Freeze all layers**
for param in model.parameters():
    param.requires_grad = False
**Unfreeze final layers**
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
** Replace classifier**
model.fc = nn.Linear(model.fc.in_features, 10)


**Data Preparation**

- Images resized, center cropped, and normalized to match ImageNet input specs.
- Dataset split: 80% training, 20% validation.

transform = transforms.Compose([...])
full_train_dataset = ImageFolder(...)
train_dataset, val_dataset = random_split(...)

----------------------------------------

** Training with wandb Logging**

Training loop logs loss and accuracy:

wandb.init(project="Assignment_2_CNN", config={...})

for epoch in range(epochs):
    model.train()
    ...
    wandb.log({"training_loss": ..., "validation_accuracy": ...})

- Uses `Adam` optimizer on unfreezed layers.
- Logs training/validation stats to wandb.


 **Results Visualization**

Loss and accuracy over epochs:

plt.plot(range(epochs), train_losses)
plt.plot(range(epochs), val_accuracies)



**Example Output**

- Training Loss vs Epochs
- Validation Accuracy vs Epochs
- wandb dashboard for detailed logs and plots



 **Requirements**

Install dependencies with:
torch
torchvision
matplotlib
wandb


 **File Structure**

.
├── train.py

├── plot_images.py

├── README.md


**Run Training**

Update `dataset_path` and run:

python train.py


The codebase can be modularized into the following structure for better maintainability:

project/

├── model.py              # Model definition and customization

├── train.py              # Main training script

├── get_dataloaders.py               # Dataset loading and transforms

├── utils.py              # Helper functions (e.g., plotting)

├── config.py             # Configuration variables

├── train_wandb           # Wandb training

└── README.md             # Project documentation

Each file can encapsulate one aspect of the pipeline, making it easier to debug, reuse, or extend.
