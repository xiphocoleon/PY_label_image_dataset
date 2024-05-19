import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets 
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt

### START VARIABLES ###
csv_images_labeled = 'C:\\Users\\thoma\\Documents\\Computer Vision\\geodis.csv'
images_dir = 'C:\\Users\\thoma\\Pictures\\Geodis all_images'
### END VARIABLES ###

### START CLASSES ###
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
     
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx,0]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
### END CLASSES ###

### START FUNCTIONS ###
def plot_with_labels(training_data):
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag", 
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8,8))
    cols, rows = 3,3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    
def show_label(image, label):
    """Show image with label"""
    plt.imshow(image)
    plt.text(x=10, y=10, s=label)
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()

### END FUNCTIONS ###

### START CREATE TEST AND TRAIN DATA FROM PRE-BUILT FASHION SET FOR TEST ###
'''
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
    )

plot_with_labels(training_data)      
'''
### END CREATE TEST AND TRAIN DATA FROM PRE-BUILT FASHION SET FOR TEST ###

### START ITERATE THROUGH DATA SAMPLES WITH CUSTOM DATASET ###
custom_dataset = CustomImageDataset(annotations_file=csv_images_labeled,
                                    img_dir=images_dir)

fig = plt.figure()

for i in range(len(custom_dataset)):
    sample = custom_dataset[i]

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_label(**sample)

    if i == 3:
        plt.show()
        break