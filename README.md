# MAP-MLF 2026 Final Project - ISEP

## Image Classification (4 classes) with PyTorch CNN

### Project Structure

```
train_v10.py                      # Main training script
y_train_v2.csv                    # Training labels (9,227 samples)
y_test_submission_example_v2.csv  # Test IDs (3,955 samples)
images/                           # 13,182 PNG images (45×51 pixels)
requirements.txt                  # Python dependencies
README.md                         # This file
```

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python train_v10.py
```

### Approach

- **Ensemble CNN:** 4 architectures (ResCNN, DeepResCNN, SEDeepResCNN, WideResCNN)
- **Validation:** 5-Fold Stratified Cross-Validation
- **Training Strategy:** 
  - Pseudo-labeling (2-round: train on fold predictions, retrain on confident test predictions)
  - 4 architectures × 5 seeds × 5 folds = 100 models per round
- **Augmentation:** CutMix (p=0.5), horizontal/vertical flips
- **Inference:** TTA with 4 geometric flips + temperature scaling (T=0.5)
- **Ensemble:** Simple averaging of top models

### Results

- Kaggle score: **0.97667**
- OOF accuracy: ~97.2%
- Training time: ~2-3 hours (5-fold CV, 2-round pseudo-labeling)

### Author

ISEP MAP-MLF Final Project 2026
# MAP-MLF 2026 Final Project - ISEP

## Classification dimages (4 classes) avec CNN PyTorch

### Structure du projet

`
train.py                          # Script principal
submission.csv                    # Soumission Kaggle
y_train_v2.csv                    # Labels train (9227)
y_test_submission_example_v2.csv  # IDs test (3955)
images/                           # 13182 images PNG 45x51
requirements.txt                  # Dependances
`

### Installation

`ash
pip install torch torchvision numpy pandas pillow scikit-learn
`

### Execution

`ash
python train.py
`

### Approche

- CNN avec blocs residuels (Conv - BN - ReLU - ResBlock - Pool)
- Ensemble 5-Fold Stratified Cross-Validation
- Data augmentation (flip H/V + bruit)
- TTA 8 passes par modele
- Label smoothing 0.05

### Resultats

- OOF accuracy: ~97%
- Kaggle score: 0.96
