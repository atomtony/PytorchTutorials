import os
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt

plt.ion()

data_dir = 'data/hymenoptera_data'

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose({
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
})

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), train_transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

train_size = len(train_dataset)
val_size = len(val_dataset)

class_names = train_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputs, classes = next(iter(train_dataloader))

out = torchvision.utils.make_grid(inputs)


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


imshow(out, title=[class_names[x] for x in classes])

print("a")
