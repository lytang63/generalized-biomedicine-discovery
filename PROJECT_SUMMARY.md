# GBD Project Summary

## Code Cleaning Completed

### Personal Information Removed
- ✅ User paths in `config.py` replaced with placeholders
- ✅ Personal paths in training scripts removed
- ✅ Specific server paths cleaned from all `.sh` files
- ✅ Hardcoded user directories replaced with generic paths

### Exploratory Code Removed
- ✅ Deleted `*_ori.py` files (original exploration versions)
- ✅ Deleted `*_old.py` files (deprecated versions)
- ✅ Deleted `*_v1.py`, `*_v3.py`, `*_v4.py`, `*_v5_v1.py` (intermediate model versions)
- ✅ Removed duplicate scripts (e.g., `contrastive_train_gastrovisioncopy.sh`)
- ✅ Removed exploration notes (`5880 selex命令.txt`, `results.txt`)
- ✅ Cleaned temporary files (`.DS_Store`, `__pycache__`, `*.pyc`)

### Documentation Created
- ✅ **README.md**: Main project documentation with full setup and usage
- ✅ **USAGE.md**: Quick start guide with examples
- ✅ **README_SelEx.md**: Original SelEx documentation (preserved for reference)

## Repository Structure

```
GBD_SelEx/
├── README.md                          # Main documentation
├── USAGE.md                          # Quick start guide
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
├── config.py                         # Configuration (cleaned paths)
│
├── contrastive_train_*.sh           # Training scripts (4 datasets)
│   ├── contrastive_train_gastrovision.sh
│   ├── contrastive_train_histoset.sh
│   ├── contrastive_train_mll23.sh
│   └── contrastive_train_derm12345.sh
│
├── data/                            # Dataset loaders
│   ├── GBD/                        # Biomedical datasets
│   │   ├── gastrovision.py
│   │   ├── histoset.py
│   │   ├── mll23.py
│   │   ├── derm12345.py
│   │   ├── chestxray14.py
│   │   ├── fitzpatrick17k.py
│   │   └── split_unusual.py        # Utility for creating splits
│   ├── get_datasets.py             # Dataset factory with mode splits
│   └── [standard GCD datasets]     # CUB, CIFAR, etc.
│
├── models/                          # Neural network architectures
│   ├── gbd_layers.py               # Bio-Active Soft HGNN (main)
│   ├── gbd_layers_v5.py            # Alternative version
│   ├── vision_transformer.py       # ViT backbone
│   └── vision_transformer2.py      # ViT backbone v2
│
├── methods/                         # Training methods
│   ├── contrastive_training/
│   │   ├── contrastive_training.py
│   │   └── contrastive_training_wandb.py  # Main training script
│   ├── clustering/                 # KMeans-based evaluation
│   └── estimate_k/                 # K estimation utilities
│
├── project_utils/                   # Utilities
│   ├── cluster_utils.py
│   ├── cluster_and_log_utils.py
│   ├── general_utils.py
│   └── schedulers.py
│
└── kmeans_pytorch_/                 # Local kmeans_pytorch installation
```

## Key Files by Function

### Configuration
- `config.py`: Dataset paths, model paths, output directories

### Training Entry Points
- `methods/contrastive_training/contrastive_training_wandb.py`: Main training script
- `contrastive_train_*.sh`: Shell scripts for each dataset (all 3 modes)

### Core Models
- `models/gbd_layers.py`: Bio-Active Hypergraph Neural Network
- `models/vision_transformer2.py`: Vision Transformer backbone with modifications

### Dataset Loaders
- `data/GBD/*.py`: Biomedical dataset loaders (4 datasets)
- `data/get_datasets.py`: Mode splits definition (lines 154-254)

## Three Discovery Settings (Modes)

Defined in `data/get_datasets.py`:

### Mode 1: Semantic Heuristic
- **Principle**: Domain knowledge drives the known/unknown split
- **GastroVision**: Normal anatomy → Pathological findings
- **HistoSet**: Common tissues → Specialized tissues
- **MLL23**: Normal cells → Abnormal cells
- **Derm12345**: Benign → Malignant patterns

### Mode 2: Long-tail Distribution
- **Principle**: Frequency-based split simulates real-world data imbalance
- **Known**: Top 50% most frequent classes (well-documented)
- **Unknown**: Bottom 50% rare classes (discovery target)

### Mode 3: Hierarchical Split
- **Principle**: Medical taxonomy guides the split
- **Known**: Selected branches from medical hierarchy
- **Unknown**: Complementary branches requiring discovery

## Four Biomedical Datasets

| Dataset | Classes | Domain | Mode 1 Known | Mode 2 Known | Mode 3 Known |
|---------|---------|--------|--------------|--------------|--------------|
| GastroVision | 27 | Endoscopy | 11 | 14 | 14 |
| HistoSet | 14 | Histopathology | 5 | 7 | 8 |
| MLL23 | 18 | Microscopy | 13 | 9 | 10 |
| Derm12345 | 40 | Dermatology | 20 | 20 | 26 |

## Usage

### Basic Training
```bash
python -m methods.contrastive_training.contrastive_training_wandb \
    --dataset_name histoset \
    --mode 1 \
    --batch_size 128 \
    --epochs 200
```

### Run All Modes for a Dataset
```bash
bash contrastive_train_histoset.sh
```

### Full Paper Experiments
```bash
for script in contrastive_train_*.sh; do
    bash "$script"
done
```

## Output

Training logs are saved to `./outputs/` with format:
```
logfile_{exp_num}_{dataset}_mode{mode}.out
```

Key metrics reported:
- **Train ACC Unlabelled_v2**: Main metric (reported in paper)
- Test accuracies: All / Old (known) / New (unknown) classes

## Dependencies

Core requirements:
- PyTorch
- torchvision
- scikit-learn
- pandas
- pillow
- kmeans_pytorch (custom install required)

Optional:
- wandb (for logging)

See `requirements.txt` and `environment.yml` for full list.

## Citation

```bibtex
@inproceedings{GBD2026,
  title={Generalized Biomedicine Discovery},
  author={[Authors]},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Relation to SelEx

This work extends [SelEx](https://github.com/SarahRastegar/SelEx) (ECCV 2024) to biomedical domain:
- Base framework: SelEx GCD methodology
- Novel contribution: Bio-Active Soft HGNN architecture
- Novel contribution: Three discovery settings for biomedical scenarios
- Novel contribution: Four biomedical benchmarks

---

**Status**: ✅ Code cleaned and ready for release
**Date**: 2026-09-09
