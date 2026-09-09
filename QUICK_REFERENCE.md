# GBD-SelEx: Quick Reference

## Installation (3 steps)
```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. kmeans_pytorch
git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch && pip install --editable . && cd ..

# 3. Download DINO weights
mkdir -p pretrained_models/dino
wget -P pretrained_models/dino https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
```

## Configuration
Edit `config.py`:
```python
gastrovision_dataroot = "path/to/Gastrovision"
histoset_dataroot = "path/to/HistoSet-5x14"
mll23_dataroot = "path/to/MLL23"
derm12345_dataroot = "path/to/derm12345"
```

## Run Experiments
```bash
# Single dataset, all 3 settings
bash contrastive_train_histoset.sh

# All paper experiments
for dataset in gastrovision histoset mll23 derm12345; do
    bash contrastive_train_${dataset}.sh
done
```

---

## Three Discovery Settings

| Setting | Known Classes | Unknown Classes | Clinical Rationale |
|---------|---------------|-----------------|-------------------|
| **I: Long-tail** | Frequent (top 50%) | Rare (bottom 50%) | Common diseases documented; rare require discovery |
| **II: Normal-Abnormal** | Normal anatomy | Pathological | Clinical workflow detects pathology vs. normal |
| **III: Taxonomy** | Known siblings | Novel siblings | New concepts emerge within diagnostic families |

---

## Four Biomedical Datasets

| Dataset | Classes | Modality | Setting I Split | Setting II Split | Setting III Split |
|---------|---------|----------|-----------------|------------------|-------------------|
| **GastroVision** | 27 | Endoscopy | 11 → 16 | 14 → 13 | 14 → 13 |
| **HistoSet** | 14 | Histopathology | 5 → 9 | 7 → 7 | 8 → 6 |
| **MLL23** | 18 | Microscopy | 13 → 5 | 9 → 9 | 10 → 8 |
| **Derm12345** | 40 | Dermatology | 20 → 20 | 20 → 20 | 26 → 14 |

---

## Key Hyperparameters (Paper Settings)

```bash
--num_hyperedges 16              # Evidence slots (E in paper)
--use_gru_update True            # Token refinement method
--batch_size 128
--epochs 200
--lr 0.1
--sup_con_weight 0.35            # Balance supervised/unsupervised
--unsupervised_smoothing 1.0
--unbalanced True                # Long-tail handling
--seed 42
```

---

## SCAN Architecture

**SCAN (Surprise-evoked Complementary AccommodatioN)**

```python
from models.gbd_layers import SCAN

scan = SCAN(
    embed_dim=768,
    num_evidence_slots=16,       # E evidence prototypes
    topk_tokens_per_slot=None,   # Information bottleneck (optional)
    use_gru_update=True
)
```

### Three-Stage Cognitive Transformation

**Stage I: Predictive Suppression**
- Build dominant anchor from evidence slots
- Filter anticipated patterns geometrically
- Output: Residual tokens (unpredicted signals)

**Stage II: Surprise-Evoked Salience**
- Quantify surprise by residual magnitude
- Amplify high-surprise tokens with salience weighting
- Aggregate salience-conditioned evidence

**Stage III: Complementary Accommodation**
- Map excited evidence to novelty vector
- Project to orthogonal subspace (reduce interference)
- Gate by surprise, knownness, peakedness
- Stable integration via residual connection

---

## Expected Results Format

Training outputs to `./outputs/logfile_{N}_{dataset}_mode{M}.out`

```
====== Reports for the best checkpoint: ======
Train ACC Unlabelled_v2:           # ← Main metric (paper)
  All: 0.XXX | Old: 0.XXX | New: 0.XXX
Test ACC:
  All: 0.XXX | Old: 0.XXX | New: 0.XXX
```

**Key Metric**: `Train ACC Unlabelled_v2 → New` = Novel class discovery accuracy

---

## Quick Test

```bash
# Verify setup
python -c "from models.gbd_layers import SCAN; print('✓ SCAN OK')"

# Run single experiment (Setting I, HistoSet)
python -m methods.contrastive_training.contrastive_training_wandb \
    --dataset_name histoset \
    --mode 1 \
    --batch_size 128 \
    --epochs 200 \
    --num_hyperedges 16 \
    --use_gru_update True \
    --seed 42
```

---

## Performance Summary

### Setting I: Long-tail (Average across 4 datasets)
- SelEx: 55.5% All | 61.7% Old | 51.8% New
- **SelEx + SCAN: 66.0% All | 77.1% Old | 54.2% New** (+10.5 All)

### Setting II: Normal-Abnormal (Average across 4 datasets)
- SelEx: 50.6% All | 58.9% Old | 47.4% New
- **SelEx + SCAN: 64.8% All | 76.4% Old | 56.3% New** (+14.2 All)

### Setting III: Within-Taxonomy (Average across 4 datasets)
- SelEx: 53.7% All | 70.0% Old | 49.8% New
- **SelEx + SCAN: 66.0% All | 83.7% Old | 53.1% New** (+12.3 All)

---

## Runtime Estimates

- **Single run** (1 dataset × 1 setting): ~3-4 hours (V100/A100)
- **Full paper** (4 datasets × 3 settings): ~36-48 hours
- **GPU Memory**: ~10-12GB with batch_size=128

---

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Dataset and model paths |
| `data/get_datasets.py` | Setting splits (lines 154-254) |
| `models/gbd_layers.py` | SCAN implementation |
| `methods/contrastive_training/contrastive_training_wandb.py` | Training loop |
| `contrastive_train_*.sh` | Run scripts per dataset |

---

## Citation

```bibtex
@inproceedings{tang2026gbd,
  title={Generalized Biomedicine Discovery},
  author={Tang, Luyao and Yang, Yingkai and Chen, Hanqi and Zheng, Jiewei and Chen, Chaoqi and Chen, Cheng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

**Status**: ✅ Ready for release  
**Paper**: "Generalized Biomedicine Discovery" (ECCV 2026)  
**Method**: SCAN (Surprise-evoked Complementary AccommodatioN)
