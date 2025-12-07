
import os
import random
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets


data_dir = "/Users/chachaen/Desktop/dataset"    
output_dir = "outputs"

num_classes = 5
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
num_epochs = 20
lr = 1e-3
val_ratio = 0.3        



def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None

set_seed(seed)


train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

full_dataset_for_split = datasets.ImageFolder(root=data_dir)
dataset_size = len(full_dataset_for_split)
val_size = max(1, int(dataset_size * val_ratio))
train_size = dataset_size - val_size

print(f"Total samples: {dataset_size}, train: {train_size}, val: {val_size}")


train_dataset, val_dataset = random_split(
    full_dataset_for_split,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)


full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

print("Class to index mapping:", full_dataset.class_to_idx)


model = models.resnet50(pretrained=True)


for param in model.parameters():
    param.requires_grad = False


in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    running_correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, running_correct / total



def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return running_loss / total, running_correct / total



best_acc = 0
best_weights = copy.deepcopy(model.state_dict())

os.makedirs(output_dir, exist_ok=True)

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_model(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")


    if val_acc > best_acc:
        best_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, f"{output_dir}/resnet_finetuned_best.pth")


model.load_state_dict(best_weights)
torch.save(model.state_dict(), f"{output_dir}/resnet_finetuned_last.pth")
print(f"\nTraining complete. Best val acc = {best_acc:.4f}")
print("Models saved to:", output_dir)


infer_dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)
idx2class = {v:k for k,v in infer_dataset.class_to_idx.items()}
label_map = {
    "1": "Poor (P)",
    "2": "Unacceptable (U)",
    "3": "Marginal (M)",
    "4": "Acceptable (A)",
    "5": "Excellent (E)"
}

model.eval()
print("\n===== Inference on all images =====")
with torch.no_grad():
    for img_path, _ in infer_dataset.samples:
        img = infer_dataset.loader(img_path)
        img = val_transforms(img).unsqueeze(0).to(device)
        out = model(img)
        pred = out.argmax(dim=1).item()

        class_name = idx2class[pred]  
        readable = label_map.get(class_name, class_name)

        print(f"{os.path.basename(img_path)} → Predicted {class_name} ({readable})")
