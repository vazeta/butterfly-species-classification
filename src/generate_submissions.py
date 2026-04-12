import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models


def submit_to_kaggle(file_path, message, competition="aca-butterflies"):
    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "submit",
                "-c", competition,
                "-f", str(file_path),
                "-m", str(message),
            ],
            capture_output=True,
            text=True,
        )
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        if result.returncode == 0:
            print(f"Submission successful: {file_path}")
        else:
            print(f"Submission failed: {file_path}")
    except Exception as e:
        print(f"Error during submission: {e}")


PATH             = "../aca-butterflies"
CHECKPOINTS_ROOT = Path("../checkpoints_CNN")
SUBMISSIONS_DIR  = Path("../submissions_CNN_report")
SEED_FOLDER_PAT  = re.compile(r"checkpoints_(\d+)$")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 75
BATCH_SIZE  = 32
IMAGE_SIZE  = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RGB_MEAN = [0.4790, 0.4646, 0.3369]
RGB_STD  = [0.2560, 0.2462, 0.2558]

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


df_train_full = pd.read_csv(os.path.join(PATH, "train.csv"))
all_classes   = sorted(df_train_full["label"].unique())
class_to_idx  = {cls: idx for idx, cls in enumerate(all_classes)}
idx_to_class  = {idx: cls for cls, idx in class_to_idx.items()}


def val_transform(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

transform_rgb      = val_transform(RGB_MEAN,      RGB_STD)
transform_imagenet = val_transform(IMAGENET_MEAN, IMAGENET_STD)


class ButterflyTestDataset(data.Dataset):
    def __init__(self, img_dir, filenames, transform=None):
        self.img_dir   = img_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.filenames[idx]


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256*6*6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(4096, 4096),    nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    def forward(self, x): return self.classifier(torch.flatten(self.features(x), 1))


class AlexNetOptimized(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),   nn.BatchNorm2d(96),  nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),  nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256*6*6, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.classifier(torch.flatten(self.features(x), 1))


class AlexNetPretrained(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        backbone = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256*6*6, 512), nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.classifier(torch.flatten(self.features(x), 1))


class VGG16(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        layers, in_ch = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.classifier(torch.flatten(self.features(x), 1))


class VGG16Pretrained(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        for p in self.features.parameters():
            p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512*7*7, 512), nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.classifier(torch.flatten(self.features(x), 1))


class ModernLeNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.fc1   = nn.Linear(64 * 56 * 56, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.bn1(self.conv1(x))), 2)
        x = torch.max_pool2d(torch.relu(self.bn2(self.conv2(x))), 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


ARCH_REGISTRY = {
    "alexnet_pre" : (AlexNetPretrained, True),
    "alexnet_opt" : (AlexNetOptimized,  False),
    "alexnet"     : (AlexNet,           False),
    "vgg16_pre"   : (VGG16Pretrained,   True),
    "vgg16"       : (VGG16,             False),
    "modernlenet" : (ModernLeNet,       False),
}

def build_model_from_run_name(stem: str) -> tuple:
    """Returns (model, transform) inferred from the checkpoint stem."""
    for key, (model_cls, use_imagenet) in ARCH_REGISTRY.items():
        if stem.startswith(key):
            transform = transform_imagenet if use_imagenet else transform_rgb
            return model_cls(num_classes=NUM_CLASSES), transform
    raise ValueError(f"Unknown architecture for stem: {stem}")


@torch.no_grad()
def generate_submission(model, transform, test_dir, output_path):
    filenames = sorted(os.listdir(test_dir))
    test_ds   = ButterflyTestDataset(test_dir, filenames, transform)
    loader    = data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model.eval()
    ids, preds_out = [], []

    for images, fnames in loader:
        images = images.to(DEVICE)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        ids.extend(fnames)
        preds_out.extend([idx_to_class[p] for p in preds])

    df_sub = pd.DataFrame({"filename": ids, "label": preds_out})
    df_sub.to_csv(output_path, index=False)
    print(f"  saved -> {output_path}  ({len(df_sub)} samples)")
    return df_sub


test_dir = os.path.join(PATH, "test")
skipped  = []

for seed_folder in sorted(CHECKPOINTS_ROOT.iterdir()):
    m = SEED_FOLDER_PAT.match(seed_folder.name)
    if not m:
        continue
    seed = m.group(1)

    for pt_path in sorted(seed_folder.glob("*.pt")):
        run_name = re.sub(r"_seed\d+$", "", pt_path.stem)
        out_path = SUBMISSIONS_DIR / f"submission_cnn_{seed}_{run_name}.csv"

        if out_path.exists():
            print(f"  skip (exists): {out_path.name}")
            continue

        print(f"[seed={seed}] {run_name}")

        try:
            model, transform = build_model_from_run_name(run_name)
            ckpt = torch.load(pt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model = model.to(DEVICE)
            generate_submission(model, transform, test_dir, out_path)
            submit_to_kaggle(file_path=out_path, message=out_path.name)

        except Exception as e:
            print(f"  ERROR: {e}")
            skipped.append((seed, run_name, str(e)))

print("\nDone.")
if skipped:
    print("Skipped / errored:")
    for s in skipped:
        print(f"  seed={s[0]}  run={s[1]}  reason={s[2]}")