# Quick Start Guide

## Installation

1. **Clone and setup environment**:
```bash
git clone <repository_url>
cd GBD_SelEx
pip install -r requirements.txt
```

2. **Install kmeans_pytorch**:
```bash
git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch
pip install --editable .
cd ..
```

3. **Download pretrained models**:
   - [DINO ViT-B/16](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)
   - [DINOv2 ViT-B/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth)
   
   Place in `pretrained_models/dino/`

4. **Configure paths** in `config.py`:
```python
# Dataset paths
gastrovision_dataroot = "path/to/Gastrovision"
histoset_dataroot = "path/to/HistoSet-5x14"
mll23_dataroot = "path/to/MLL23"
derm12345_dataroot = "path/to/derm12345"

# Pretrained model paths
dino_pretrain_path = 'pretrained_models/dino/dino_vitbase16_pretrain.pth'
dino_pretrain_path2 = 'pretrained_models/dino/dinov2_vitb14_reg4_pretrain.pth'
```

## Training Examples

### Single experiment
```bash
python -m methods.contrastive_training.contrastive_training_wandb \
    --dataset_name histoset \
    --mode 1 \
    --batch_size 128 \
    --epochs 200 \
    --num_hyperedges 16 \
    --seed 42
```

### All three modes for a dataset
```bash
bash contrastive_train_histoset.sh
```

### All datasets (full paper experiments)
```bash
bash contrastive_train_gastrovision.sh
bash contrastive_train_histoset.sh
bash contrastive_train_mll23.sh
bash contrastive_train_derm12345.sh
```

## Understanding the Three Modes

| Mode | Known Classes | Unknown Classes | Clinical Rationale |
|------|---------------|-----------------|-------------------|
| 1 | Normal/Anatomical | Pathological | Start with normal anatomy |
| 2 | Common (head) | Rare (tail) | Well-documented vs. novel |
| 3 | Taxonomy-based | Complementary | Medical hierarchies |

### Dataset-specific Mode Definitions

**GastroVision** (27 classes):
- Mode 1: 11 normal/anatomical classes → 16 pathological unknown
- Mode 2: 14 most frequent classes → 13 rare unknown
- Mode 3: 14 taxonomy-based → 13 unknown

**HistoSet** (14 classes):
- Mode 1: 5 common tissue types → 9 unknown
- Mode 2: 7 most frequent → 7 unknown
- Mode 3: 8 hierarchy-based → 6 unknown

**MLL23** (18 classes):
- Mode 1: 13 normal cell types → 5 abnormal unknown
- Mode 2: 9 most frequent → 9 rare unknown
- Mode 3: 10 lineage-based → 8 unknown

**Derm12345** (40 classes):
- Mode 1: 20 head classes → 20 tail unknown
- Mode 2: 20 alternative split → 20 unknown
- Mode 3: 26 balanced split → 14 unknown

## Expected Results

Training will output:
- **Train ACC Unlabelled**: Accuracy on unlabeled training data (reported metric)
- **Test ACC**: Accuracy on test set
- Breakdown: All / Old (known) / New (unknown) classes

Look for:
```
Reports for the best checkpoint:
  Train ACC Unlabelled_v2: All/Old/New accuracies
  Test ACC: All/Old/New accuracies

Reports for the last checkpoint:
  [same format]
```

## Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--mode` | Discovery setting (1/2/3) | Try all three |
| `--num_hyperedges` | Hypergraph capacity | 16 |
| `--use_gru_update` | Dynamic hypergraph | True |
| `--unsupervised_smoothing` | Label smoothing | 1.0 |
| `--sup_con_weight` | Supervised loss weight | 0.35 |
| `--unbalanced` | Long-tail handling | True |
| `--seed` | Random seed | 42 |

## Output Structure

```
outputs/
├── logfile_1_histoset_mode1.out
├── logfile_2_histoset_mode2.out
├── logfile_3_histoset_mode3.out
└── ...
```

Each log contains:
- Training progress (loss, accuracy per epoch)
- Evaluation metrics (every N epochs)
- Final results for best and last checkpoints

## Troubleshooting

**CUDA out of memory**:
- Reduce `--batch_size` (try 64 or 32)
- Reduce `--num_workers`

**Dataset not found**:
- Check paths in `config.py`
- Verify CSV files exist in dataset root

**kmeans_pytorch issues**:
- Reinstall with `pip install --editable .`
- Alternative: modify code to use sklearn KMeans (results may differ)

**Slow training**:
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check `CUDA_VISIBLE_DEVICES` is set correctly

## Citation

```bibtex
@inproceedings{GBD2026,
  title={Generalized Biomedicine Discovery},
  author={[Authors]},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
