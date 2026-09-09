# Generalized Biomedicine Discovery (GBD)

Official PyTorch implementation of **"Generalized Biomedicine Discovery"** (ECCV 2026).

**Authors**: Luyao Tang¹, Yingkai Yang², Hanqi Chen², Jiewei Zheng³, Chaoqi Chen², and Cheng Chen†¹  
¹ The University of Hong Kong, ² Shenzhen University, ³ Xiamen University

---

## Abstract

In real-world clinical practice, medical images face open-world shifts: (i) long-tailed rare diseases, (ii) subtle lesions dominated by normal anatomy, and (iii) hierarchical taxonomies. Yet most open-world paradigms assume flat, balanced label spaces, leaving these biomedical demands unresolved. 

We introduce **Generalized Biomedicine Discovery (GBD)** and a unified benchmark spanning long-tail, anomaly, and taxonomy-aware discovery. Our key insight is that dominant known patterns form a visual manifold that masks subtle novelty. Inspired by expert diagnosis, we propose **SCAN (Surprise-evoked Complementary AccommodatioN)**, which follows a cognition-inspired perceptual progression: it applies predictive suppression to filter expected norms, triggers surprise-evoked salience to highlight unexpected deviations, and performs complementary accommodation to integrate these shifts into global representations.

**Keywords**: Generalized biomedicine discovery · Generalized category discovery · Medical image analysis · Cognitive science

---

## Key Contributions

1. **GBD Benchmark**: A clinically grounded open-world paradigm with three realistic settings:
   - **Setting I**: Long-tail Rare Disease Discovery
   - **Setting II**: Normal-to-Abnormal Discovery  
   - **Setting III**: Within-Taxonomy Discovery

2. **SCAN Method**: A plug-and-play cognitive vision layer implementing:
   - **Stage I (Predictive Suppression)**: Filters anticipated patterns under predictive coding
   - **Stage II (Surprise-Evoked Salience)**: Uses residual energy to amplify unexpected deviations
   - **Stage III (Complementary Accommodation)**: Integrates novelty while preserving established structure

3. **Strong Results**: Extensive experiments show SCAN improves conventional GCD methods on the GBD benchmark, with particularly strong gains for SelEx in isolating subtle anomalies and discovering long-tailed rare diseases.

---

## Datasets

We evaluate on four biomedical imaging datasets across diverse modalities:

| Dataset | Classes | Modality | Domain |
|---------|---------|----------|---------|
| **GastroVision** | 27 | Endoscopy | Upper/lower GI tract findings |
| **HistoSet-5×14** | 14 | Histopathology | Multi-organ H&E (5 organs) |
| **MLL23** | 18 | Microscopy | Peripheral blood cells (hematopoietic) |
| **DERM12345** | 40 | Dermatology | Skin lesions (5 super-classes) |

### Dataset Preparation

Update dataset paths in `config.py`:

```python
gastrovision_dataroot = "path/to/Gastrovision"
histoset_dataroot = "path/to/HistoSet-5x14"
mll23_dataroot = "path/to/MLL23"
derm12345_dataroot = "path/to/derm12345"
```

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Pre-trained Models

Download DINO pre-trained weights:

```bash
mkdir -p pretrained_models/dino
cd pretrained_models/dino

# DINOv2 ViT-B/14 (default backbone used in paper)
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth

cd ../..
```

Update paths in `config.py`:

```python
dino_pretrain_path2 = 'pretrained_models/dino/dinov2_vitb14_reg4_pretrain.pth'
```

### Verify Installation

```bash
python -c "from models.gbd_layers import SCAN; print('✓ SCAN import OK')"
python -c "from data.get_datasets import get_datasets; print('✓ Dataset loaders OK')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
```

---

## Three Discovery Settings

### Setting I: Long-tail Rare Disease Discovery

Models frequent diseases as known classes and discovers rare tail diseases.

- **Rationale**: Clinical annotations focus on common conditions; rare but critical diseases must be discovered
- **Split**: Known = top 50% most frequent classes; Novel = bottom 50% (tail)

### Setting II: Normal-to-Abnormal Discovery

Models normal anatomy as known classes and discovers heterogeneous abnormalities.

- **Rationale**: Clinical workflow detects pathology relative to normal baselines
- **Split**: Known = normal anatomical variants; Novel = pathological findings

### Setting III: Within-Taxonomy Discovery

Models known diagnostic families and discovers novel sibling subtypes within established hierarchies.

- **Rationale**: Medical knowledge is hierarchically organized; new concepts emerge as fine-grained siblings
- **Split**: Known = subset of siblings per taxonomy node; Novel = remaining siblings

---

## Training

### Quick Start

Train SCAN on a specific dataset and setting:

```bash
python -m methods.contrastive_training.contrastive_training_wandb \
    --dataset_name histoset \
    --mode 1 \
    --batch_size 128 \
    --epochs 200 \
    --base_model vit_dino \
    --num_workers 8 \
    --sup_con_weight 0.35 \
    --weight_decay 5e-5 \
    --unsupervised_smoothing 1.0 \
    --lr 0.1 \
    --eval_funcs v1 v2 \
    --num_hyperedges 16 \
    --use_gru_update True \
    --seed 42
```

**Note**: `--num_hyperedges` controls the number of evidence slots (E=16 in paper).

### Using Shell Scripts

We provide training scripts for all four datasets:

```bash
# Run all 3 modes for HistoSet
bash contrastive_train_histoset.sh

# Run full paper experiments
bash contrastive_train_gastrovision.sh
bash contrastive_train_histoset.sh
bash contrastive_train_mll23.sh
bash contrastive_train_derm12345.sh
```

### Key Hyperparameters

| Parameter | Description | Paper Value |
|-----------|-------------|-------------|
| `--mode` | Discovery setting (1/2/3) | 1, 2, or 3 |
| `--num_hyperedges` | Number of evidence slots (E) | 16 |
| `--use_gru_update` | Token refinement with GRU | True |
| `--batch_size` | Training batch size | 128 |
| `--epochs` | Total training epochs | 200 |
| `--lr` | Initial learning rate | 0.1 |
| `--sup_con_weight` | Supervised contrastive loss weight | 0.35 |
| `--unsupervised_smoothing` | Label smoothing for unlabeled data | 1.0 |
| `--unbalanced` | Handle long-tailed distributions | True |
| `--seed` | Random seed | 42 |

---

## Evaluation

The training script automatically evaluates on both validation and test sets using Hungarian matching.

### Metrics

- **All**: Accuracy across all classes (known + novel)
- **Old**: Accuracy on known classes
- **New**: Accuracy on novel (discovered) classes

### Reading Results

Look for these sections in `outputs/logfile_*.out`:

```
====== Reports for the best checkpoint: ======
Train ACC Unlabelled_v2:           # ← Main metric (Hungarian matching)
  All: 0.XXX | Old: 0.XXX | New: 0.XXX
Test ACC:
  All: 0.XXX | Old: 0.XXX | New: 0.XXX
```

**Key Metric**: `Train ACC Unlabelled_v2 → New` corresponds to novel class discovery accuracy reported in the paper.

---

## Method Overview: SCAN

**SCAN (Surprise-evoked Complementary AccommodatioN)** is a lightweight plug-and-play layer that operates on vision backbone tokens.

### Three-Stage Architecture

```python
from models.gbd_layers import SCAN

# Instantiate SCAN
scan = SCAN(
    embed_dim=768,
    num_evidence_slots=16,       # E in paper (evidence prototypes)
    topk_tokens_per_slot=None,   # Information bottleneck (optional)
    use_gru_update=True          # Token refinement method
)

# Forward pass
class_token, patch_tokens = backbone(image)  # [B,D], [B,N,D]
updated_class_token = scan(patch_tokens, class_token)
```

### Stage I: Predictive Suppression

Filters anticipated patterns to expose residual signals:

1. Route patch tokens into E evidence slots
2. Form sample-conditioned **dominant anchor** (**u** in paper)
3. Geometrically project out predictable components: **r**_i = **x**_i - (**x**_i^T **u**)**u**

### Stage II: Surprise-Evoked Salience

Amplifies unexpected, localized deviations:

1. Quantify **surprise energy** s_i = ||**r**_i||₂ (residual magnitude)
2. Convert to **salience weights** via softmax over surprise
3. Re-aggregate residuals weighted by both slot assignment and surprise

### Stage III: Complementary Accommodation

Integrates novelty without disrupting known structure:

1. Map excited evidence to global **novelty vector** (Δ in paper)
2. Project to **complementary subspace**: Δ_⊥ = Δ - (Δ^T **u**)**u**
3. Gate by surprise (S), knownness (k), and peakedness (κ)
4. Stable accommodation: **c**~ = **c** + γ · LN(Δ_⊥ + **x**_pool⊥)

**No extra loss is introduced** — SCAN is trained end-to-end with the host GCD objective.

---

## Results Summary

### Setting I: Long-tail Rare Disease Discovery (Average across 4 datasets)

| Method | All | Old | New |
|--------|-----|-----|-----|
| SelEx | 55.5 | 61.7 | 51.8 |
| **SelEx + SCAN** | **66.0** (+10.5) | **77.1** (+15.4) | **54.2** (+2.4) |

### Setting II: Normal-to-Abnormal Discovery (Average across 4 datasets)

| Method | All | Old | New |
|--------|-----|-----|-----|
| SelEx | 50.6 | 58.9 | 47.4 |
| **SelEx + SCAN** | **64.8** (+14.2) | **76.4** (+17.5) | **56.3** (+8.9) |

### Setting III: Within-Taxonomy Discovery (Average across 4 datasets)

| Method | All | Old | New |
|--------|-----|-----|-----|
| SelEx | 53.7 | 70.0 | 49.8 |
| **SelEx + SCAN** | **66.0** (+12.3) | **83.7** (+13.7) | 53.1 (+3.3) |

**Key Takeaway**: SCAN consistently improves discovery performance, with particularly strong gains on known (Old) classes and overall accuracy, demonstrating effective novelty integration without catastrophic forgetting.

---

## Project Structure

```
GBD_SelEx/
├── config.py                    # Dataset and model paths
├── requirements.txt             # Python dependencies
├── contrastive_train_*.sh      # Training scripts (4 datasets)
│
├── data/
│   ├── GBD/                    # Biomedical dataset loaders
│   │   ├── gastrovision.py     # GastroVision (27 classes)
│   │   ├── histoset.py         # HistoSet (14 classes)
│   │   ├── mll23.py            # MLL23 (18 classes)
│   │   └── derm12345.py        # DERM12345 (40 classes)
│   └── get_datasets.py         # Dataset factory with mode splits
│
├── models/
│   ├── gbd_layers.py           # SCAN (main implementation)
│   ├── gbd_layers_v5.py        # SCANv5 (alternative version)
│   └── vision_transformer2.py  # ViT backbone wrappers
│
├── methods/
│   └── contrastive_training/
│       └── contrastive_training_wandb.py  # Main training loop
│
└── project_utils/              # Clustering and evaluation utilities
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tang2026gbd,
  title={Generalized Biomedicine Discovery},
  author={Tang, Luyao and Yang, Yingkai and Chen, Hanqi and Zheng, Jiewei and Chen, Chaoqi and Chen, Cheng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

## Acknowledgements

This codebase builds upon:
- [SelEx](https://github.com/SarahRastegar/SelEx) (ECCV 2024) - Base GCD framework
- [GCD](https://github.com/sgvaze/generalized-category-discovery) - Original generalized category discovery

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or issues, please open an issue in this repository.
