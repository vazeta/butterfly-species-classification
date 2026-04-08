import os
import sys
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed when running as .py
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from dataset_utilsCNN import ButterflyDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../support"))
from submission_script import submit_to_kaggle

# =============================================================================
# CONFIG
# =============================================================================

SEEDS = [0, 1, 1010, 42]

PATH        = "../aca-butterflies"
NUM_CLASSES = 75
BATCH_SIZE  = 32
IMAGE_SIZE  = 224
NUM_EPOCHS  = 30
PATIENCE    = 7

RGB_MEAN = [0.4790, 0.4646, 0.3369]
RGB_STD  = [0.2560, 0.2462, 0.2558]

# since we are using pre trained NN we decided to normalize it using this values 
# discussion of value origin can be found here https://github.com/pytorch/vision/pull/1965
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# =============================================================================
# TRANSFORMS
# =============================================================================

def build_transforms(mean, std, image_size=IMAGE_SIZE):
    train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    return train, val

# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=NUM_EPOCHS, patience=PATIENCE, device=device, run_name="experiment"):
    
    counter_early_stopper = 0
    best_acc = 0.0
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    
    history = {"train_loss": [], "val_loss": [], "train_acc":  [], "val_acc":  [], "lr": []}

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)
        
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, *_ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}")
        
        if vl_acc > best_acc + 1e-4:
            best_acc = vl_acc
            best_loss = vl_loss
            best_weights = copy.deepcopy(model.state_dict())
            counter_early_stopper = 0
        elif vl_loss < best_loss - 1e-4:
            best_loss = vl_loss
            counter_early_stopper = 0
        else:
            counter_early_stopper += 1
            if counter_early_stopper >= patience:
                print(f"\n[Early Stopping] {run_name} parado na época {epoch}")
                break

    elapsed_in_min = (time.time() - t0) / 60
    model.load_state_dict(best_weights)
    return model, history, elapsed_in_min

# =============================================================================
# METRICS & PLOTS
# =============================================================================

def compute_metrics(labels, preds, name=""):
    acc = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro",  zero_division=0)
    f1_wei = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"{name}->Accuracy : {acc:.4f}")
    print(f"{name}->F1 (macro) : {f1_mac:.4f}")
    print(f"{name}->F1(weighted) : {f1_wei:.4f}")
    return {"accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wei}


def plot_history(history, title="Training History", save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

# =============================================================================
# MODELS
# =============================================================================

class AlexNetPretrained(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNetPretrained, self).__init__()
        
        backbone = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        self.features = backbone.features
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG16Pretrained(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VGG16Pretrained, self).__init__()
        
        backbone = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        
        self.features = backbone.features
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =============================================================================
# CRITERION & OPTIMIZER
# =============================================================================

def build_optimizer(name, params, lr, weight_decay=1e-4):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

def build_criterion(name):
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif name == "multi_margin":
        return nn.MultiMarginLoss(margin=1.0)

# =============================================================================
# EXPERIMENTS — top-3 models from seed 2026
# =============================================================================
# last parameter -> flag_pretrained
experiments = [
    ("alexnet_pre_cel_adam",    lambda: AlexNetPretrained(), "cross_entropy", "adam",    1e-4, 30, 1),
    ("vgg16_pre_cel_adam",      lambda: VGG16Pretrained(),   "cross_entropy", "adam",    1e-4, 30, 1),
    ("vgg16_pre_cel_rmsprop",   lambda: VGG16Pretrained(),   "cross_entropy", "rmsprop", 1e-4, 30, 1),
    ("alexnet_pre_cel_rmsprop", lambda: AlexNetPretrained(), "cross_entropy", "rmsprop", 1e-4, 30, 1),
]

# =============================================================================
# TEST DATASET (used for per-run submissions)
# =============================================================================

class TestDataset(data.Dataset):
    def __init__(self, filenames, img_dir, transform=None):
        self.filenames = filenames
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# =============================================================================
# MAIN LOOP
# =============================================================================

all_results    = []
best_f1        = -1.0
best_ckpt_info = {}

for SEED in SEEDS:
    print(f"\n{'#'*70}")
    print(f"# SEED {SEED}")
    print(f"{'#'*70}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_transform, val_transform = build_transforms(RGB_MEAN, RGB_STD)
    train_transform_pre_trained, val_transform_pre_trained = build_transforms(IMAGENET_MEAN, IMAGENET_STD)


    img_dir = os.path.join(PATH, "train")
    df = pd.read_csv(os.path.join(PATH, "train.csv"))

    df_train, df_val = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)

    all_classes  = sorted(df["label"].unique())
    idx_to_class = {cls: idx for idx, cls in enumerate(all_classes)}

    train_dataset = ButterflyDataset(df_train, img_dir, train_transform, idx_to_class)
    val_dataset   = ButterflyDataset(df_val,   img_dir, val_transform,   idx_to_class)

    train_dataset_pretrained = ButterflyDataset(df_train, img_dir, train_transform_pre_trained, idx_to_class)
    val_dataset_pretrained = ButterflyDataset(df_val, img_dir, val_transform_pre_trained, idx_to_class)

    train_loader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )

    train_loader_pretrained = data.DataLoader(
    train_dataset_pretrained, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0,            # Alterar para 0 para evitar Broken pipe
    pin_memory=False,         # Correto: Manter False para Memória Unificada
    # persistent_workers e prefetch_factor não são necessários com num_workers=0
)

    val_loader_pretrained = data.DataLoader(
        val_dataset_pretrained, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,            # Alterar para 0
        pin_memory=False
    )

    results_summary = []
    trained_models  = {}
    all_histories   = {}
    all_times       = {}

    for run_name, model_fn, loss_name, optim_name, lr, epochs, flag_pretrained in experiments:
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"Loss: {loss_name}  |  Optimizer: {optim_name}  |  LR: {lr}")
        print(f"{'='*60}")

        run_device = device
        print(f"Executando em: {run_device}")

        model     = model_fn().to(run_device)
        criterion = build_criterion(loss_name)
        optimizer = build_optimizer(optim_name, model.parameters(), lr=lr)

        if flag_pretrained:
            train_loader_use = train_loader_pretrained
            val_loader_use = val_loader_pretrained
        else:
            train_loader_use = train_loader
            val_loader_use = val_loader
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        model, history, time_var = train_model(
            model, train_loader_use, val_loader_use, 
            criterion, optimizer, 
            scheduler=scheduler, 
            device=run_device, 
            run_name=run_name,
            num_epochs=epochs
        )

        _, _, labels, preds = evaluate(model, val_loader_use, criterion, run_device)
        metrics = compute_metrics(labels, preds, name=run_name)

        results_summary.append({
            "run"      : run_name,
            "seed"     : SEED,
            "loss"     : loss_name,
            "optimizer": optim_name,
            "lr"       : lr,
            "epochs"   : epochs,
            "time_min" : round(time_var, 2),
            **metrics
        })
        all_results.append(results_summary[-1])

        trained_models[run_name] = model
        all_histories[run_name]  = history
        all_times[run_name]      = time_var

        checkpoint_path = f"../checkpoints_CNN/checkpoints_{SEED}"
        os.makedirs(checkpoint_path, exist_ok=True)

        ckpt_file = f"{checkpoint_path}/{run_name}_seed{SEED}.pt"
        torch.save({
            "model_state": model.state_dict(), 
            "history"    : history,
            "metrics"    : metrics
        }, ckpt_file)

        plot_history(history, title=run_name,
                     save_path=f"{checkpoint_path}/{run_name}_seed{SEED}_history.png")

        # Track best checkpoint globally
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_ckpt_info = {
                "run_name" : run_name,
                "seed"     : SEED,
                "ckpt_path": ckpt_file,
                "model_fn" : model_fn,
                "loss_name": loss_name,
            }

        # Submit this run to Kaggle
        df_full      = pd.read_csv(os.path.join(PATH, "train.csv"))
        all_classes  = sorted(df_full["label"].unique())
        idx_to_class = {cls: idx for idx, cls in enumerate(all_classes)}
        int_to_class = {v: k for k, v in idx_to_class.items()}

        _, val_transform_sub = build_transforms(RGB_MEAN, RGB_STD)
        test_dir       = os.path.join(PATH, "test")
        test_filenames = sorted(os.listdir(test_dir))

        test_dataset = TestDataset(test_filenames, test_dir, val_transform_sub)
        test_loader_sub = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=0, pin_memory=False)

        run_preds = []
        with torch.no_grad():
            for images in test_loader_sub:
                images  = images.to(run_device)
                outputs = model(images)
                preds   = outputs.argmax(dim=1).cpu().numpy()
                run_preds.extend([int_to_class[p] for p in preds])

        submission_dir  = "../submissions_CNN"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = f"{submission_dir}/{run_name}_seed_{SEED}_submission.csv"
        pd.DataFrame({"filename": test_filenames, "label": run_preds}).to_csv(submission_path, index=False)
        print(f"Submission saved: {submission_path}")

        submit_to_kaggle(
            file_path=submission_path,
            message=f"{run_name}->{SEED} F1={metrics['f1_macro']:.4f}"
        )

    print("\n TEMPOS DE EXECUÇÃO:")
    for key, value in all_times.items():
        print(f"{key} -> {value:.2f} min")

# =============================================================================
# FULL RESULTS SUMMARY
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv("../checkpoints_CNN/seed_experiments_results.csv", index=False)
print("\n\nFULL RESULTS:")
print(results_df.to_string(index=False))

# =============================================================================
# SEED 2026 — load existing checkpoints, generate submissions, compare
# =============================================================================

ckpt_2026_dir   = "../checkpoints_CNN/checkpoints_2026"
relevant_2026   = {
    "alexnet_pre_cel_adam"   : lambda: AlexNetPretrained(),
    "vgg16_pre_cel_adam"     : lambda: VGG16Pretrained(),
    "vgg16_pre_cel_rmsprop"  : lambda: VGG16Pretrained(),
    "alexnet_pre_cel_rmsprop": lambda: AlexNetPretrained(),
}

df_full      = pd.read_csv(os.path.join(PATH, "train.csv"))
all_classes  = sorted(df_full["label"].unique())
idx_to_class = {cls: idx for idx, cls in enumerate(all_classes)}
int_to_class = {v: k for k, v in idx_to_class.items()}
_, val_transform_2026 = build_transforms(RGB_MEAN, RGB_STD)
test_dir       = os.path.join(PATH, "test")
test_filenames = sorted(os.listdir(test_dir))

if os.path.exists(ckpt_2026_dir):
    for run_name, model_fn in relevant_2026.items():
        ckpt_path = os.path.join(ckpt_2026_dir, f"{run_name}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {ckpt_path} not found")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        f1    = ckpt["metrics"]["f1_macro"]
        print(f"[Seed 2026] {run_name} -> F1-macro: {f1:.4f}")

        # Generate submission for this checkpoint
        model_2026 = model_fn().to(device)
        model_2026.load_state_dict(ckpt["model_state"])
        model_2026.eval()

        test_dataset_2026 = TestDataset(test_filenames, test_dir, val_transform_2026)
        test_loader_2026  = data.DataLoader(test_dataset_2026, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=0, pin_memory=False)
        run_preds = []
        with torch.no_grad():
            for images in test_loader_2026:
                images  = images.to(device)
                outputs = model_2026(images)
                preds   = outputs.argmax(dim=1).cpu().numpy()
                run_preds.extend([int_to_class[p] for p in preds])

        submission_dir  = "../submissions_CNN"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = f"{submission_dir}/{run_name}_seed2026_submission.csv"
        pd.DataFrame({"filename": test_filenames, "label": run_preds}).to_csv(submission_path, index=False)
        print(f"Submission saved: {submission_path}")

        submit_to_kaggle(
            file_path=submission_path,
            message=f"{run_name} seed=2026 F1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_ckpt_info = {
                "run_name" : run_name,
                "seed"     : 2026,
                "ckpt_path": ckpt_path,
                "model_fn" : model_fn,
                "loss_name": "cross_entropy",
            }
else:
    print(f"[INFO] {ckpt_2026_dir} not found, skipping seed 2026")

print(f"\nBest overall: {best_ckpt_info['run_name']} "
      f"(seed={best_ckpt_info['seed']}, F1-macro={best_f1:.4f})")