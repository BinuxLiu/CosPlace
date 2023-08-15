
import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors


def open_image(path):
    return Image.open(path).convert("RGB")


class TeachDataset(data.Dataset):
    def __init__(self, dataset_folder):
        """
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.dataset_name = os.path.basename(dataset_folder) # Output: train
        
        subfolders = [f.name for f in os.scandir(self.dataset_folder) if f.is_dir()]
        for folder_name in subfolders:
            os.makedirs(self.dataset_folder.split("_d")[0] + "_feat/" + folder_name, exist_ok=True)
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        #### Read paths and UTM coordinates for all images.
        self.images_paths = sorted(glob(os.path.join(self.dataset_folder, "**", "*.jpg"), recursive=True))

        self.images_num = len(self.images_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)
        normalized_img = self.base_transform(pil_img)
        return normalized_img, image_path, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #db: {self.images_num} >"
