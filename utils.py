# Imports
import os
import random
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Reproductabilidade
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Path
try:
    import kagglehub
    kagglehub.login()
    path = kagglehub.competition_download('aca-butterflies')
    print("Path to dataset files:", path)

except ImportError as e:
    from pathlib import Path
    path = Path().cwd() / "aca-butterflies"

# Constantes
BATCH_SIZE = 32
IMAGE_SIZE = 64

# Valores RGB (calculados no 0_EDA.ipynb)
RGB_MEAN = [0.4790, 0.4646, 0.3369]
RGB_STD = [0.2560, 0.2462, 0.2558]

# Transform
data_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# Dataset
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


# Carregamento — dataset completo (usado no EDA)
img_dir = os.path.join(path, 'train')
df = pd.read_csv(os.path.join(path, 'train.csv'))

dataset = ButterflyDataset(df=df, img_dir=img_dir, transform=data_transform)
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(dataset.classes)

# Split estratificado 80/20
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

# Datasets de treino e validação
train_dataset = ButterflyDataset(
    df=train_df, img_dir=img_dir, transform=data_transform
)
val_dataset = ButterflyDataset(
    df=val_df, img_dir=img_dir, transform=data_transform
)

# shuffle=True: ordem aleatória a cada epoch (evita padrões espúrios)
train_loader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
# shuffle=False: ordem fixa — na validação não se aprende, só se avalia
val_loader = data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)
