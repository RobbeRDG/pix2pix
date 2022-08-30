import math
from random import random
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset

import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)


    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        # Get the image path
        img_file_path = os.path.join(self.root_dir, self.list_files[idx])

        # Get the full image file as an np array
        # The full image is always 1200 x 600 (2 parts)
        img = np.array(Image.open(img_file_path))

        # Extract the imput and target image
        input_img = img[:, :600, :]
        target_img = img[:, 600:, :]

        # Apply augmentations
        # Global
        augmentations = config.both_transform(image = input_img, image0 = target_img)
        input_img, target_img = augmentations["image"], augmentations["image0"]

        # Input only
        input_img = config.transform_only_input(image=input_img)["image"]

        # Target only
        target_img = config.transform_only_mask(image=target_img)["image"]

        return input_img, target_img


def test_dataset():
    dataset = MapDataset("data/train/")

    # Test length
    print("length: ", len(dataset))

    # Test get image
    input, target = dataset.__getitem__(3)
    plt.subplot(1,2,1)
    plt.imshow(np.dstack(input), interpolation='none', aspect='auto')
    plt.axis('off')
    plt.title("input")

    plt.subplot(1,2,2)
    plt.imshow(np.dstack(target), interpolation='none', aspect='auto')
    plt.axis('off')
    plt.title("target")

    plt.savefig("test.png")

if __name__ == "__main__":
    test_dataset()

