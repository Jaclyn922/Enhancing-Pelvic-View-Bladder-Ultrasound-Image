import torch
from torch.utils.data import Dataset
import glob, re, os, collections
from PIL import Image

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_data(img_dir, cls_labels):
    all_images, all_labels = [], []
    print('loading data ...')

    for lb in cls_labels:

        all_imgs = sorted(
            glob.glob(os.path.join(img_dir, lb, "*.png")),
            key=numericalSort
        )

        for img_path in all_imgs:
            all_images.append(img_path)       
            all_labels.append(int(lb) - 1)     

    print('total images:', len(all_images))
    counter = collections.Counter(all_labels)
    print('class images:', collections.OrderedDict(sorted(counter.items())))
    return all_images, all_labels


def collate_fn(batch):
    imgs_batch, labels_batch = zip(*batch)
    imgs_batch = torch.stack(imgs_batch)
    labels_batch = torch.stack(labels_batch)
    return imgs_batch, labels_batch


class USQDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images       
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.images[idx]      
        img = Image.open(img_path).convert("RGB") 

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.int64)

        return img, label
