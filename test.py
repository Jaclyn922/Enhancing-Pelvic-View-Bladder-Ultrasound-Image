import numpy as np
import matplotlib.pyplot as plt

import random
import os
from collections import namedtuple
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import argparse
from dataloader import load_data, USQDataset, collate_fn
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


# ============================================================
# Auto JPG → PNG converter
# ============================================================
def convert_jpg_to_png(data_dir):
    classes = ['1', '2', '3', '4', '5']
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)

            if fname.lower().endswith(('.jpg', '.jpeg')):
                img = Image.open(fpath).convert("RGB")
                new_name = fname.rsplit('.', 1)[0] + ".png"
                new_path = os.path.join(cls_path, new_name)
                img.save(new_path)
                os.remove(fpath)
                print(f"[Converted] {fpath} → {new_path}")




class Wrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        logits = self.net(x)
        return (logits,)   


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--data_dir', dest="data_dir", default="dataset/")
    parser.add_argument('--height', dest="height", default=224, type=int)
    parser.add_argument('--width', dest="width", default=224, type=int)
    args = parser.parse_args()
    return args


def main(args):
    convert_jpg_to_png(args.data_dir)

    cls_labels = ['1', '2', '3', '4', '5']
    test_images, test_labels = load_data(img_dir=args.data_dir, cls_labels=cls_labels)

    print("Total images loaded:", len(test_images))


    transform_val = transforms.Compose([
        transforms.Resize((224,224)),   
        transforms.ToTensor(),         
    ])


    test_ds = USQDataset(test_images, test_labels, transform=transform_val)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = "cpu"
    print("Using device:", device)


    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 5)
    net = net.to(device)
    net.eval()

    print("Loaded ResNet-18 baseline with pretrained ImageNet weights.")

    wrapped = Wrapper(net)


    print("Running a quick debug forward on first batch to check shapes/types...")
    try:
        it = iter(test_dl)
        inputs, labels = next(it)
        print("DEBUG: inputs type:", type(inputs), "labels type:", type(labels))
        print("DEBUG: inputs shape:", getattr(inputs, "shape", None))
        with torch.no_grad():
            outputs = wrapped(inputs.to(device))
        print("DEBUG: model outputs type:", type(outputs))
        if isinstance(outputs, tuple):
            print("DEBUG: outputs[0] shape:", outputs[0].shape)
        else:
            print("DEBUG: outputs shape:", outputs.shape)
    except StopIteration:
        print("DEBUG: test_dl is empty (no batches).")
    except Exception as e:
        print("DEBUG: Exception during debug forward:", repr(e))


    test_model(wrapped, test_dl, cls_labels, device, save_dir=".")


if __name__ == "__main__":
    args = parse()
    main(args)
