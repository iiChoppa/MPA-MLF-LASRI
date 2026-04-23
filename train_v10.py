"""
MAP-MLF 2026 Final Project - ISEP
V10: Built on V7 (Kaggle 0.97667) — the proven best
Strategy: More diversity + Pseudo-labeling (2-round training)

Changes vs V7:
- 4th architecture: WideResCNN (wider channels from start: 64->128->256)
- Pseudo-labeling Round 2: retrain with confident Round 1 test predictions
  → effectively increases training data
- 4 architectures × 5 seeds × 5 folds = 100 models per round
- TTA = 4 (flips only — NO rotations, images are 45x51 not square!)
- Simple averaging (proven, no score-weighting)
- NO SWA (proven harmful in V9)
- NO Mixup (proven harmful in V8)

Mapping: id -> img_{id+1}.png
"""
import os, sys, time, gc
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(DATA_DIR, 'train_v10.log')
IMG_DIR = os.path.join(DATA_DIR, 'images')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    line = str(msg)
    sys.stdout.write(line + '\n')
    sys.stdout.flush()
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

with open(LOG_PATH, 'w') as f:
    f.write('')

# === Config ===
N_FOLDS = 5
MAX_EPOCHS = 120
PATIENCE = 25
BATCH_SIZE = 128
TTA_PASSES = 4          # Flips only (proven, no rotations on non-square images)
LABEL_SMOOTHING = 0.05   # V7 proven value
PSEUDO_CONF = 0.995      # Confidence threshold for pseudo-labeling

log(f"PyTorch {torch.__version__}, Device: {DEVICE}")

# ============================================================
# DATA LOADING
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'y_train_v2.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'y_test_submission_example_v2.csv'))

def load_all_images(ids):
    imgs = []
    for img_id in ids:
        path = os.path.join(IMG_DIR, f'img_{img_id + 1}.png')
        img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
        imgs.append(np.transpose(img, (2, 0, 1)))
    return np.array(imgs)

log("Loading images...")
t0 = time.time()
X_all = load_all_images(train_df['id'].values)
y_all = train_df['target'].values
X_test_np = load_all_images(test_df['id'].values)
log(f"  Train: {X_all.shape}, Test: {X_test_np.shape} ({time.time()-t0:.1f}s)")

MEAN = X_all.mean(axis=(0, 2, 3), keepdims=True).reshape(1, 3, 1, 1)
STD = X_all.std(axis=(0, 2, 3), keepdims=True).reshape(1, 3, 1, 1)
X_all = (X_all - MEAN) / (STD + 1e-7)
X_test_np = (X_test_np - MEAN) / (STD + 1e-7)

N_CLASSES = len(np.unique(y_all))
log(f"  Classes: {N_CLASSES}, distribution: {np.bincount(y_all)}")

# ============================================================
# DATA AUGMENTATION (V5/V7 style - simple & proven)
# ============================================================
def augment_batch(X):
    X = X.clone()
    mask_h = torch.rand(X.size(0), device=X.device) > 0.5
    X[mask_h] = X[mask_h].flip(3)
    mask_v = torch.rand(X.size(0), device=X.device) > 0.5
    X[mask_v] = X[mask_v].flip(2)
    return X

# ============================================================
# BLOCKS
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

# ============================================================
# ARCHITECTURE A: ResCNN (32->64->128)
# ============================================================
class ResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            ResBlock(32),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, N_CLASSES)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ============================================================
# ARCHITECTURE B: DeepResCNN (32->64->128->256)
# ============================================================
class DeepResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            ResBlock(32),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, N_CLASSES)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ============================================================
# ARCHITECTURE C: SE-DeepResCNN (channel attention)
# ============================================================
class SEDeepResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            ResBlock(32), SEBlock(32),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), SEBlock(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), SEBlock(128),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256), SEBlock(256),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, N_CLASSES)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

# ============================================================
# ARCHITECTURE D: WideResCNN (wider channels, fewer pooling stages)
# Starts at 64 channels (vs 32), uses 2 ResBlocks per stage
# ============================================================
class WideResCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), ResBlock(64),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), ResBlock(128),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            ResBlock(256), ResBlock(256),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, N_CLASSES)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ============================================================
# TRAIN ONE FOLD (V7 recipe — proven)
# ============================================================
def train_fold(model_class, X_tr, y_tr, X_vl, y_vl, fold_name, lr):
    log(f"\n--- {fold_name} (train={len(X_tr)}, val={len(X_vl)}) ---")
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(TensorDataset(X_vl, y_vl), batch_size=512)

    model = model_class().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_val, best_state, wait = 0, None, 0

    for ep in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            X = augment_batch(X)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in vl_loader:
                correct += (model(X.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
                total += len(y)
        val_acc = correct / total
        scheduler.step()
        dt = time.time() - t0

        improved = val_acc > best_val
        if improved:
            best_val, wait = val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if (ep + 1) % 10 == 0 or ep == 0 or improved:
            log(f"  E{ep + 1:3d}: val={val_acc:.4f} best={best_val:.4f} ({dt:.0f}s){' *' if improved else ''}")

        if wait >= PATIENCE:
            log(f"  Early stop at epoch {ep + 1}")
            break

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  => Best: {best_val:.4f}")
    return best_state, best_val

# ============================================================
# TTA PREDICTION (4 passes — flips only, proven safe)
# ============================================================
def predict_tta(model_class, state_dict, X_test):
    model = model_class().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    results = []
    for i in range(TTA_PASSES):
        X = torch.from_numpy(X_test.astype(np.float32))
        if i == 1:
            X = X.flip(3)          # horizontal flip
        elif i == 2:
            X = X.flip(2)          # vertical flip
        elif i == 3:
            X = X.flip(2).flip(3)  # both flips (= 180°)
        probs = []
        with torch.no_grad():
            for (b,) in DataLoader(TensorDataset(X), batch_size=512):
                probs.append(torch.softmax(model(b.to(DEVICE)), 1).cpu().numpy())
        results.append(np.concatenate(probs))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.mean(results, axis=0)

# ============================================================
# ARCHITECTURE CONFIGS
# ============================================================
ARCHS = [
    # Architecture A: ResCNN (compact)
    ("ResCNN",       ResCNN,       1.5e-3, 42),
    ("ResCNN",       ResCNN,       1.5e-3, 123),
    ("ResCNN",       ResCNN,       1.5e-3, 7),
    ("ResCNN",       ResCNN,       1.5e-3, 2024),
    ("ResCNN",       ResCNN,       1.5e-3, 999),
    # Architecture B: DeepResCNN (deep)
    ("DeepResCNN",   DeepResCNN,   1e-3,   42),
    ("DeepResCNN",   DeepResCNN,   1e-3,   123),
    ("DeepResCNN",   DeepResCNN,   1e-3,   7),
    ("DeepResCNN",   DeepResCNN,   1e-3,   2024),
    ("DeepResCNN",   DeepResCNN,   1e-3,   999),
    # Architecture C: SEDeepResCNN (deep + attention)
    ("SEDeepResCNN", SEDeepResCNN, 8e-4,   42),
    ("SEDeepResCNN", SEDeepResCNN, 8e-4,   123),
    ("SEDeepResCNN", SEDeepResCNN, 8e-4,   7),
    ("SEDeepResCNN", SEDeepResCNN, 8e-4,   2024),
    ("SEDeepResCNN", SEDeepResCNN, 8e-4,   999),
    # Architecture D: WideResCNN (wide + double ResBlocks) — NEW
    ("WideResCNN",   WideResCNN,   8e-4,   42),
    ("WideResCNN",   WideResCNN,   8e-4,   123),
    ("WideResCNN",   WideResCNN,   8e-4,   7),
    ("WideResCNN",   WideResCNN,   8e-4,   2024),
    ("WideResCNN",   WideResCNN,   8e-4,   999),
]

# ############################################################
# ROUND 1: Standard training
# ############################################################
total_models = len(ARCHS) * N_FOLDS
log(f"\n{'=' * 60}")
log(f"ROUND 1: {len(ARCHS)} configs x {N_FOLDS} folds = {total_models} models")
log(f"{'=' * 60}")

r1_models = []
r1_scores = []
t_start = time.time()

for arch_name, model_class, lr, seed in ARCHS:
    log(f"\n{'='*40}")
    log(f"R1 | {arch_name} | LR={lr} | Seed={seed}")
    log(f"{'='*40}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr = torch.from_numpy(X_all[tr_idx].astype(np.float32))
        y_tr = torch.from_numpy(y_all[tr_idx].astype(np.int64))
        X_vl = torch.from_numpy(X_all[vl_idx].astype(np.float32))
        y_vl = torch.from_numpy(y_all[vl_idx].astype(np.int64))

        state, score = train_fold(model_class, X_tr, y_tr, X_vl, y_vl,
                                  f"{arch_name} S{seed} F{fold+1}/{N_FOLDS}", lr)
        r1_models.append((model_class, state))
        r1_scores.append(score)
        gc.collect()
        torch.cuda.empty_cache()

t_r1 = time.time() - t_start
log(f"\nRound 1 complete ({t_r1 / 60:.1f} min)")
log(f"  Mean CV: {np.mean(r1_scores):.4f}")

# Per-architecture summary
idx = 0
for arch_name, _, _, seed in ARCHS:
    scores_for = r1_scores[idx:idx+N_FOLDS]
    log(f"  {arch_name} S{seed}: {[f'{s:.4f}' for s in scores_for]} mean={np.mean(scores_for):.4f}")
    idx += N_FOLDS

for arch_type in ["ResCNN", "DeepResCNN", "SEDeepResCNN", "WideResCNN"]:
    arch_scores = []
    for i, (name, _, _, _) in enumerate(ARCHS):
        if name == arch_type:
            arch_scores.extend(r1_scores[i*N_FOLDS:(i+1)*N_FOLDS])
    log(f"  >> {arch_type} overall mean: {np.mean(arch_scores):.4f}")

# ============================================================
# ROUND 1 TEST PREDICTIONS
# ============================================================
log(f"\nRound 1 test predictions ({len(r1_models)} models + TTA{TTA_PASSES})...")
r1_probs = []
for i, (mcls, sd) in enumerate(r1_models):
    if (i + 1) % 10 == 0 or i == 0:
        log(f"  Model {i + 1}/{len(r1_models)} TTA...")
    r1_probs.append(predict_tta(mcls, sd, X_test_np))

r1_ensemble = np.mean(r1_probs, axis=0)
r1_preds = r1_ensemble.argmax(axis=1)
r1_conf = r1_ensemble.max(axis=1)

# Save Round 1 submission (this is already a good submission — equivalent to V7+)
sub_r1 = pd.DataFrame({'id': test_df['id'].values, 'target': r1_preds})
sub_r1.to_csv(os.path.join(DATA_DIR, 'submission_r1.csv'), index=False)
log(f"\nsubmission_r1.csv saved (Round 1, {len(r1_preds)} predictions)")

unique, counts = np.unique(r1_preds, return_counts=True)
log("Round 1 distribution:")
for u, c in zip(unique, counts):
    log(f"  Class {u}: {c} ({100 * c / len(r1_preds):.1f}%)")

# ############################################################
# PSEUDO-LABELING: Select high-confidence test predictions
# ############################################################
pseudo_mask = r1_conf >= PSEUDO_CONF
n_pseudo = pseudo_mask.sum()
log(f"\n{'=' * 60}")
log(f"PSEUDO-LABELING: {n_pseudo}/{len(r1_conf)} test samples (conf >= {PSEUDO_CONF})")
log(f"{'=' * 60}")

if n_pseudo < 50:
    log("Too few pseudo-labels, skipping Round 2. Using Round 1 as final.")
    # Just save Round 1 as final
    sub = pd.DataFrame({'id': test_df['id'].values, 'target': r1_preds})
    sub.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    log(f"submission.csv saved ({len(r1_preds)} predictions) [Round 1 only]")
else:
    pseudo_X = X_test_np[pseudo_mask]
    pseudo_y = r1_preds[pseudo_mask]
    log(f"  Pseudo-label distribution: {np.bincount(pseudo_y, minlength=N_CLASSES)}")
    log(f"  Mean confidence of selected: {r1_conf[pseudo_mask].mean():.4f}")

    # Combine original training data with pseudo-labeled data
    X_combined = np.concatenate([X_all, pseudo_X], axis=0)
    y_combined = np.concatenate([y_all, pseudo_y], axis=0)
    log(f"  Combined training set: {len(X_combined)} ({len(X_all)} real + {n_pseudo} pseudo)")

    # ############################################################
    # ROUND 2: Retrain with pseudo-labels
    # ############################################################
    log(f"\n{'=' * 60}")
    log(f"ROUND 2: Retrain with pseudo-labels")
    log(f"  {len(ARCHS)} configs x {N_FOLDS} folds = {total_models} models")
    log(f"{'=' * 60}")

    r2_models = []
    r2_scores = []
    t_r2_start = time.time()

    for arch_name, model_class, lr, seed in ARCHS:
        log(f"\n{'='*40}")
        log(f"R2 | {arch_name} | LR={lr} | Seed={seed}")
        log(f"{'='*40}")

        torch.manual_seed(seed + 10000)  # Different seed for Round 2
        np.random.seed(seed + 10000)
        # Use only real labels for stratified split, pseudo-labels always in train
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + 10000)

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_all, y_all)):
            # Validation: only real data (no pseudo-labels in validation!)
            X_vl = torch.from_numpy(X_all[vl_idx].astype(np.float32))
            y_vl = torch.from_numpy(y_all[vl_idx].astype(np.int64))

            # Training: real train fold + ALL pseudo-labels
            X_tr_real = X_all[tr_idx]
            y_tr_real = y_all[tr_idx]
            X_tr_combined = np.concatenate([X_tr_real, pseudo_X], axis=0)
            y_tr_combined = np.concatenate([y_tr_real, pseudo_y], axis=0)

            X_tr = torch.from_numpy(X_tr_combined.astype(np.float32))
            y_tr = torch.from_numpy(y_tr_combined.astype(np.int64))

            state, score = train_fold(model_class, X_tr, y_tr, X_vl, y_vl,
                                      f"{arch_name} S{seed} F{fold+1}/{N_FOLDS}", lr)
            r2_models.append((model_class, state))
            r2_scores.append(score)
            gc.collect()
            torch.cuda.empty_cache()

    t_r2 = time.time() - t_r2_start
    log(f"\nRound 2 complete ({t_r2 / 60:.1f} min)")
    log(f"  Mean CV: {np.mean(r2_scores):.4f}")

    # Per-architecture summary for Round 2
    idx = 0
    for arch_name, _, _, seed in ARCHS:
        scores_for = r2_scores[idx:idx+N_FOLDS]
        log(f"  {arch_name} S{seed}: {[f'{s:.4f}' for s in scores_for]} mean={np.mean(scores_for):.4f}")
        idx += N_FOLDS

    for arch_type in ["ResCNN", "DeepResCNN", "SEDeepResCNN", "WideResCNN"]:
        arch_scores = []
        for i, (name, _, _, _) in enumerate(ARCHS):
            if name == arch_type:
                arch_scores.extend(r2_scores[i*N_FOLDS:(i+1)*N_FOLDS])
        log(f"  >> {arch_type} overall mean: {np.mean(arch_scores):.4f}")

    # ============================================================
    # ROUND 2 TEST PREDICTIONS
    # ============================================================
    log(f"\nRound 2 test predictions ({len(r2_models)} models + TTA{TTA_PASSES})...")
    r2_probs = []
    for i, (mcls, sd) in enumerate(r2_models):
        if (i + 1) % 10 == 0 or i == 0:
            log(f"  Model {i + 1}/{len(r2_models)} TTA...")
        r2_probs.append(predict_tta(mcls, sd, X_test_np))

    r2_ensemble = np.mean(r2_probs, axis=0)
    r2_preds = r2_ensemble.argmax(axis=1)

    # ============================================================
    # FINAL ENSEMBLE: Combine Round 1 + Round 2
    # ============================================================
    log(f"\nFinal ensemble: Round 1 ({len(r1_probs)} models) + Round 2 ({len(r2_probs)} models)")
    all_probs_combined = r1_probs + r2_probs
    final_probs = np.mean(all_probs_combined, axis=0)
    final = final_probs.argmax(axis=1)

    # Agreement analysis
    r1_vs_r2 = (r1_preds == r2_preds).mean()
    r1_vs_final = (r1_preds == final).mean()
    log(f"  R1 vs R2 agreement: {r1_vs_r2:.4f}")
    log(f"  R1 vs Final agreement: {r1_vs_final:.4f}")

    unique, counts = np.unique(final, return_counts=True)
    log("\nFinal distribution:")
    for u, c in zip(unique, counts):
        log(f"  Class {u}: {c} ({100 * c / len(final):.1f}%)")

    # Save final submission (R1+R2 combined)
    sub = pd.DataFrame({'id': test_df['id'].values, 'target': final})
    sub.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
    log(f"\nsubmission.csv saved ({len(final)} predictions) [R1+R2 ensemble]")

    # Also save R2-only submission
    sub_r2 = pd.DataFrame({'id': test_df['id'].values, 'target': r2_preds})
    sub_r2.to_csv(os.path.join(DATA_DIR, 'submission_r2.csv'), index=False)
    log(f"submission_r2.csv saved (Round 2 only)")

    # Also save simple R1 as backup
    sub_simple = pd.DataFrame({'id': test_df['id'].values, 'target': r1_preds})
    sub_simple.to_csv(os.path.join(DATA_DIR, 'submission_simple.csv'), index=False)
    log(f"submission_simple.csv saved (Round 1 only, backup)")

log(f"\nR1 Mean CV: {np.mean(r1_scores):.4f}")
if 'r2_scores' in dir():
    log(f"R2 Mean CV: {np.mean(r2_scores):.4f}")
log(f"Total time: {(time.time() - t_start) / 60:.1f} min")
log("DONE!")
