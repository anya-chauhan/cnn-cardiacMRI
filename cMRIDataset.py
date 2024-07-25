import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class cMRIDataset(Dataset):
    def __init__(self, df, img_dir='fake1000', split='train', transform=None):
        self.df = df
        self.split = split
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(int(row['idx'])) + ".png")
        label = torch.tensor(row['sphericity_index'], dtype=torch.float32)
        image = (read_image(img_path) / 255.0).repeat(3, 1, 1)
        if self.transform:
            image = self.transform(image)
        return image, label
