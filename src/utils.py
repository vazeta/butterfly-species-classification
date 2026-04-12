import os
import random
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path = "../aca-butterflies"

BATCH_SIZE = 32
IMAGE_SIZE = 64

# Valores RGB (calculados no 0_EDA.ipynb)
RGB_MEAN = [0.4790, 0.4646, 0.3369]
RGB_STD = [0.2560, 0.2462, 0.2558]

data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


class ButterflyDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.img_labels['label'].unique())
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_name = self.img_labels.iloc[idx]['label']
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


img_dir = os.path.join(path, 'train')
df = pd.read_csv(os.path.join(path, 'train.csv'))

dataset = ButterflyDataset(df=df, img_dir=img_dir, transform=data_transform)
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(dataset.classes)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

train_dataset = ButterflyDataset(
    df=train_df, img_dir=img_dir, transform=data_transform
)
val_dataset = ButterflyDataset(
    df=val_df, img_dir=img_dir, transform=data_transform
)

train_loader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
