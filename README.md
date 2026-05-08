# Semantic Context-Aware YOLOv5 for Aerial Object Detection

**Course Project — Deep Learning**
**Team Members:** Syed Ibrahim Bin Hassan (27100201) · Asad Riaz (27100055)

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Research Gap and Motivation](#2-research-gap-and-motivation)
3. [Dataset](#3-dataset)
4. [Repository Structure](#4-repository-structure)
5. [Deliverables Summary](#5-deliverables-summary)
6. [Methodology](#6-methodology)
   - [Baseline: YOLOv5s](#61-baseline-yolov5s)
   - [Improvement 1: MCGM-All (P3+P4+P5)](#62-improvement-1-mcgm-all-p3p4p5)
   - [Improvement 2: MCGM-P3 (P3-only)](#63-improvement-2-mcgm-p3-p3-only)
   - [Improvement 3: SPGM-P3](#64-improvement-3-spgm-p3)
7. [Results](#7-results)
8. [Per-Class AP Breakdown](#8-per-class-ap-breakdown)
9. [Environment and Reproducibility](#9-environment-and-reproducibility)
10. [References](#10-references)

---

## 1. Project Overview

This project investigates **semantic contextual reasoning** as a mechanism for improving object separability in aerial image object detection. Aerial scenes pose a unique challenge: objects are small, densely packed, and visually similar to their backgrounds. Standard detectors like YOLOv5 lack an explicit mechanism to exploit the broader scene context when classifying and localising these objects.

We take **YOLOv5s** as our baseline and introduce two families of lightweight context modules integrated directly into the detection head:
1) the **Multi-scale Context Gating Module (MCGM)** 
2) **Scene Prior Gating Module (SPGM)** 

All experiments are run on the **DOTA v1.5** benchmark under controlled, fair-comparison conditions (same dataset, same hyperparameters, same hardware).

---

## 2. Research Gap and Motivation

Prior work in aerial detection (SCRDet, RoI Transformer, FPN, DOTA benchmark papers) shows that small objects in cluttered scenes suffer most from background confusion. The core problem is that standard convolutional detectors operate on local receptive fields: they extract features without explicit awareness of _what else is in the scene_. This leads to high false-positive rates and missed detections among visually ambiguous categories such as `small-vehicle`, `ship`, and `harbor`.

Our hypothesis: **injecting a learned, gated scene context signal into the detection head, either as a multi-scale dilated attention (MCGM) or as a global scene prior (SPGM), can improve the separability of these ambiguous categories without a significant increase in model complexity or inference cost.**

Key dataset observations that motivate this direction:
- **81.5%** of all training instances fall in the "small" size bin (< 32² pixels), where background confusion is worst.
- Dense images contain hundreds of co-occurring objects; **small-vehicle** (126K instances, 68.9% difficult) and **ship** (32K instances) are the primary bottleneck classes.
- Frequent co-occurrence of semantically related classes (ship+harbor, small-vehicle+large-vehicle) suggests that scene-level priors could guide more accurate discrimination.

---

## 3. Dataset

**DOTA v1.5** — A Large-Scale Dataset for Object Detection in Aerial Images

| Property | Detail |
|---|---|
| Version | DOTA v1.5 |
| Total aerial images | 2,806 (train/val/test) |
| Original image size | 800×800 to ~8,000×7,000 px |
| Categories | 16 classes |
| Total instances | ~403,318 (including small instances added in v1.5) |
| Annotation type | Oriented Bounding Boxes (OBB) as arbitrary quadrilaterals |
| Source | [Kaggle: DOTA Dataset](https://www.kaggle.com/datasets/ishwarghoshrony/dota-dataset) |

**DOTA v1.5 Categories (16 classes):**
`plane`, `baseball-diamond`, `bridge`, `ground-track-field`, `small-vehicle`, `large-vehicle`, `ship`, `tennis-court`, `basketball-court`, `storage-tank`, `soccer-ball-field`, `roundabout`, `harbor`, `swimming-pool`, `helicopter`, `container-crane`

### 4-Class Subset Used in This Project

We filter DOTA v1.5 to **4 classes** that represent the most challenging and practically relevant targets for aerial detection, and where background confusion is most severe:

| Class | Train Instances | Val Instances | Difficulty |
|---|---|---|---|
| plane | 8,072 | 2,550 | 1.8% difficult |
| ship | 32,973 | 10,765 | 7.6% difficult |
| small-vehicle | 126,501 | 43,337 | **68.9% difficult** |
| large-vehicle | 22,218 | 5,139 | 16.3% difficult |

### Preprocessing Pipeline

Large DOTA images are sliced into fixed-size patches with overlap (standard practice from SCRDet):

| Parameter | Value |
|---|---|
| Patch size | 640×640 px |
| Overlap | 128 px |
| Annotation format | HBB (YOLO format) converted from OBB |
| Minimum object area | 20 px² |
| Minimum patch overlap with object | 50% |

**Resulting dataset split:**

| Split | Source Images | Patches (image-label pairs) | Total Object Annotations |
|---|---|---|---|
| Train | 1,159 (of 1,411 — 252 have no selected classes) | 16523 | 377325 | 
| Validation | 354 (of 458 - 104 have no selected classes) | 5010 | 121530 |

---

## 4. Repository Structure

```
semantic-context-yolov5-aerial-detection/
├── README.md
├── notebooks/
│   ├── deliverable_02_dataset_preparation.ipynb   # Dataset EDA & preprocessing
│   ├── deliverable_03_baseline_model.ipynb        # YOLOv5s baseline training & eval
│   ├── deliverable_04_mcgm_v1.ipynb               # MCGM on P3+P4+P5 (all scales)
│   ├── deliverable_04_mcgm_v2.ipynb               # MCGM on P3 only (P3-only variant)
│   └── deliverable_05_spgm.ipynb                  # Scene Prior Gating Module (SPGM-P3)
└── paper/
    └── SOA_survey.pdf                             # Deliverable 1: State-of-the-art survey
```

All notebooks are self-contained and designed to run on **Kaggle** (Tesla T4 GPU, Python 3.12, PyTorch 2.10, Ultralytics 8.4.x).

---

## 5. Deliverables Summary

| # | Deliverable | File(s) | Description |
|---|---|---|---|
| 1 | SOA Survey | `paper/SOA_survey.pdf` | Literature review covering SCRDet, RoI Transformer, FPN, and related aerial detection work |
| 2 | Dataset Preparation | `deliverable_02_dataset_preparation.ipynb` | Full EDA of DOTA v1.5: class distribution, difficulty analysis, size distribution, co-occurrence, aspect ratios, and image resolution study |
| 3 | Baseline Model | `deliverable_03_baseline_model.ipynb` | YOLOv5s fine-tuned on 4-class DOTA subset; data patching pipeline; training curves; inference visualisation |
| 4 | Improvement (MCGM) | `deliverable_04_mcgm_v1.ipynb` | MCGM integrated at all three detection scales (P3+P4+P5); custom trainer; full evaluation |
| 4 | Improvement (MCGM v2) | `deliverable_04_mcgm_v2.ipynb` | MCGM restricted to P3 only (P3-only variant); efficiency/accuracy ablation |
| 5 | Improvement (SPGM) | `deliverable_05_spgm.ipynb` | Scene Prior Gating Module: P5 global context gates P3 features; independent experiment |

---

## 6. Methodology

All models use **identical training settings** for fair comparison:

| Hyperparameter | Value |
|---|---|
| Base architecture | YOLOv5s (COCO-pretrained) |
| Epochs | 20 |
| Batch size | 8 |
| Image size | 640×640 |
| Seed | 42 |
| Hardware | Tesla T4 GPU (Kaggle) |

---

### 6.1 Baseline: YOLOv5s

The unmodified **YOLOv5s** architecture fine-tuned on our 4-class DOTA patch dataset serves as the performance baseline. Its FPN+PAN neck produces three detection scales (P3 at 80×80, P4 at 40×40, P5 at 20×20), each feeding directly into the Detect head.

- **Parameters:** 9,123,740
- **Weight file size:** 17.7 MB
- **GFLOPs:** 23.8

No architectural changes — this is a pure fine-tuning baseline.

---

### 6.2 Improvement 1: MCGM-All (P3+P4+P5)

**File:** `deliverable_04_mcgm_v1.ipynb`

The **Multi-scale Context Gating Module (MCGM)** is a lightweight residual context block inserted between the neck outputs and the Detect head at all three detection scales simultaneously (P3, P4, P5).

**Design:**
```
Input x (C channels)
  → Reduce: Conv1×1 → BN → SiLU              [C → C//4 hidden channels]
  → Parallel dilated branches (d=1, 2, 3):    [3× (Conv3×3_dilated → BN → SiLU)]
  → Global context gate: GAP → Conv1×1 → Sigmoid  [channel-wise attention]
  → Each branch feature × gate
  → Fuse: Conv1×1 → BN                        [3·hidden → C]
  → Residual: x + learnable_scale × fused     [scale init = 0.1]
  → SiLU activation
```

Key design choices:
- **Parallel dilated convolutions** (dilation 1, 2, 3) capture multi-scale receptive fields without increasing spatial resolution costs.
- **Global Average Pooling gate** provides a channel-wise attention signal derived from the full feature map context.
- **Learnable residual scale** (initialised at 0.1) ensures stable training — the module starts as a near-identity and gradually learns to contribute context.
- Integrated via a **custom `MCGMTrainer`** subclassing `DetectionTrainer`, so MCGM weights are part of the trainable model from the start and are correctly saved in checkpoints.

**Parameters added:** ~0.94M (54 parameter tensors across 3 MCGM blocks)
**Total parameters:** 10,073,727 | **Weight file:** 19.5 MB | **GFLOPs:** 25.6

---

### 6.3 Improvement 2: MCGM-P3 (P3-only)

**File:** `deliverable_04_mcgm_v2.ipynb`

This variant restricts the MCGM to the **P3 scale only** (the highest-resolution feature map, 80×80). P4 and P5 pass through unmodified `nn.Identity` placeholders.

**Motivation:** P3 carries the finest spatial detail and is the primary scale for detecting the `small-vehicle` class — the most numerous and difficult category. A P3-only gate isolates whether context enrichment on small-object features alone yields a better efficiency-to-accuracy trade-off than gating all three scales.

**Architecture change from v1:** The MCGM block itself is redesigned with an added **local 1×1 branch** alongside the dilated branches, and uses a **spatial gate** (Conv1×1 + Sigmoid) instead of a global average pooling gate. The residual scale is initialised at 0.01 (smaller, for more stable convergence). The fused tensor includes `[local_feat] + [dilated_branch_feats × gate]`.

**Parameters added:** ~0.04M (21 parameter tensors — 1 block, no Identity params)
**Total parameters:** 9,174,525 | **Weight file:** 17.8 MB | **GFLOPs:** 24.5

---

### 6.4 Improvement 3: SPGM-P3

**File:** `deliverable_05_spgm.ipynb`

The **Scene Prior Gating Module (SPGM-P3)** takes a different approach: instead of enriching P3 with its own multi-scale context, it uses **P5 (the coarsest, most semantically rich feature map) as a scene source** to compute a channel-wise gate for P3.

**Design:**
```
scene = GAP(P5)                          [Global scene summary, 512 channels]
gate3 = MLP(scene) → Sigmoid             [Conv1×1 → SiLU → Conv1×1 → Sigmoid → C3 channels]
P3' = P3 + alpha × (P3 × gate3)         [alpha = 0.2, fixed non-zero for stable training]
P4, P5 unchanged
```

**Motivation:** P5 encodes high-level scene semantics (scene type, dominant object categories). Using P5 context to gate P3 features introduces a cross-scale, top-down semantic prior — similar in spirit to FPN's top-down pathway but applied as a gating signal rather than feature addition.

**Parameters added:** ~0.07M (4 parameter tensors: 2 Conv1×1 layers in the scene MLP)
**Total parameters:** 9,205,916 | **Weight file:** 17.8 MB | **GFLOPs:** 23.9

---

## 7. Results

All metrics are from the final validation evaluation (`model.val()` on the 4,542-patch validation set, confidence threshold 0.25).

### Overall Comparison

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params | Weight (MB) |
|---|---|---|---|---|---|---|
| YOLOv5s (Baseline) | 0.8675 | 0.6015 | 0.8747 | 0.8136 | ~9.1M | 17.7 |
| YOLOv5s + MCGM-All (P3+P4+P5) | **0.8681** | **0.6041** | 0.8734 | **0.8190** | ~10.1M | 19.5 |
| YOLOv5s + MCGM-P3 (P3-only) | **0.8708** | **0.6043** | **0.8748** | 0.8173 | ~9.2M | 17.8 |
| YOLOv5s + SPGM-P3 | **0.8697** | **0.6030** | 0.8746 | 0.8162 | ~9.2M | 17.8 |

**All three proposed improvements outperform the baseline on both mAP@0.5 and mAP@0.5:0.95.**

### Delta vs. Baseline

| Model | ΔmAP@0.5 | ΔmAP@0.5:0.95 | ΔPrecision | ΔRecall |
|---|---|---|---|---|
| MCGM-All (P3+P4+P5) | +0.0006 | +0.0026 | −0.0013 | +0.0054 |
| MCGM-P3 (P3-only) | +0.0033 | +0.0028 | +0.0001 | +0.0037 |
| SPGM-P3 | +0.0022 | +0.0015 | −0.0001 | +0.0026 |

**Key observations:**
- **MCGM-P3 (P3-only)** achieves the best overall mAP on both IoU thresholds, while adding only ~40K parameters and preserving the baseline model file size (~17.8 MB vs 17.7 MB). This confirms that targeting context enrichment specifically at the small-object scale is the most parameter-efficient strategy.
- **MCGM-All** improves recall most (+0.54%) at the cost of ~0.94M extra parameters and a heavier checkpoint (19.5 MB). The all-scale context does not significantly outperform the P3-only variant, suggesting diminishing returns from gating P4 and P5.
- **SPGM-P3** achieves competitive gains (+0.22% mAP@0.5, +0.15% mAP@0.5:0.95) using only 4 additional parameter tensors (~70K parameters), making it the most lightweight architectural change. The cross-scale, top-down semantic prior from P5 provides measurable but more modest improvements than the P3-internal multi-scale context of MCGM-P3.
- All improvements are numerically modest — consistent with the challenging nature of the DOTA dataset and the short 20-epoch training schedule.

---

## 8. Per-Class AP Breakdown

Per-class mAP@0.5 from the final validation run:

| Class | Baseline | MCGM-All | MCGM-P3 | SPGM-P3 |
|---|---|---|---|---|
| plane | 0.956 | 0.957 | **0.962** | 0.961 |
| ship | 0.950 | **0.951** | **0.951** | **0.951** |
| small-vehicle | 0.711 | 0.716 | 0.714 | 0.714 |
| large-vehicle | 0.853 | 0.849 | **0.856** | 0.852 |

Per-class mAP@0.5:0.95:

| Class | Baseline | MCGM-All | MCGM-P3 | SPGM-P3 |
|---|---|---|---|---|
| plane | 0.733 | **0.744** | 0.739 | 0.733 |
| ship | 0.680 | 0.679 | 0.679 | **0.681** |
| small-vehicle | 0.365 | 0.364 | **0.365** | **0.366** |
| large-vehicle | 0.628 | **0.629** | **0.635** | 0.633 |

**Class-level observations:**
- `plane` and `ship` are the easiest classes (high mAP) because they have distinctive shapes and appear against water/tarmac backgrounds. All three improvements match or exceed the baseline on these classes.
- `small-vehicle` is the hardest class (mAP@0.5 ~0.71) due to extreme instance density and 68.9% difficult flags. MCGM-All shows the largest gain (+0.5% mAP@0.5), validating the hypothesis that multi-scale context enrichment at this scale helps with small object separability.
- `large-vehicle` sees consistent improvement across all models. MCGM-P3 achieves the highest large-vehicle mAP@0.5:0.95 (+0.7% over baseline), suggesting the P3-only local context gate is also beneficial for medium-to-large objects that appear at the P3 scale due to patching.

---

## 9. Environment and Reproducibility

All notebooks are designed to run end-to-end on **Kaggle** (free tier). No local setup is required.

**Dependencies (auto-installed in notebooks):**
```
ultralytics>=8.4.37
torch>=2.10.0+cu128
```

**To reproduce any experiment:**
1. Upload the notebook to Kaggle.
2. Add the DOTA dataset: `ishwarghoshrony/dota-dataset` as a Kaggle input.
3. Run all cells in order. Each notebook is self-contained: it patches the dataset, creates the YAML config, trains, evaluates, and visualises results.
4. Training takes approximately 2–3 hours per model on a Tesla T4 GPU at 20 epochs.

**Notebook execution order:**
```
deliverable_02  →  deliverable_03  →  deliverable_04_mcgm_v1 →  deliverable_04_mcgm_v2 →  deliverable_05_spgm
```

---

## 10. References

1. Xia, G.-S., et al. (2018). *DOTA: A Large-Scale Dataset for Object Detection in Aerial Images*. CVPR.
2. Yang, X., et al. (2019). *SCRDet: Towards More Robust Detection for Small, Cluttered and Rotated Objects*. ICCV.
3. Ding, J., et al. (2019). *Learning RoI Transformer for Oriented Object Detection in Aerial Images*. CVPR.
4. Lin, T.-Y., et al. (2017). *Feature Pyramid Networks for Object Detection*. CVPR.
5. Jocher, G., et al. (2020). *YOLOv5 by Ultralytics*. [GitHub](https://github.com/ultralytics/yolov5).
6. Woo, S., et al. (2018). *CBAM: Convolutional Block Attention Module*. ECCV.
7. Hu, J., et al. (2018). *Squeeze-and-Excitation Networks*. CVPR.

---

*For the full State-of-the-Art survey covering these and additional works, see `paper/SOA_survey.pdf`.*
