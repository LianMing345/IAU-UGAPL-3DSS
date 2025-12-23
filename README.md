# Inconsistency-Aware Active Learning for Semi-supervised 3D Point Cloud Semantic Segmentation
[![IEEE TRANSACTIONS ON MULTIMEDIA](https://img.shields.io/badge/Paper-IEEE%20TMM%202025-blue)](https://ieeexplore.ieee.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%201.8.0-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.6-green)](https://www.python.org/)

## 🌟 Overview
This repository implements an **inconsistency-aware active semi-supervised learning framework** for 3D point cloud semantic segmentation. The core goal is to achieve near-fully-supervised performance with extremely limited labeled data (as low as 0.014% of training data), by deeply integrating active learning (AL) and semi-supervised learning (SSL) with novel inconsistency-aware modules.

### Key Contributions
1. **Inconsistency-Aware Uncertainty (IAU)**: Enhances active learning sample selection by fusing model uncertainty with teacher-student prediction inconsistencies, prioritizing hard-to-classify points from ambiguous/rare categories.
2. **Uncertainty-Guided Adaptive PseudoLabeling (UGAPL)**: Dynamically adjusts pseudo-label weights using AL-derived uncertainty and real-time model confidence, mitigating noise in pseudo-labels and fully utilizing unlabeled data.
3. **Deep AL-SSL Synergy**: Achieves state-of-the-art performance on indoor (S3DIS, ScanNet) and outdoor (SemanticKITTI) datasets with minimal annotations, outperforming existing methods under ultra-low annotation budgets.

## 📊 Performance
| Dataset       | Annotation Budget | mIoU (%) | Fully-Supervised Baseline (%) |
|---------------|-------------------|----------|--------------------------------|
| S3DIS (Indoor) | 0.02%             | 59.0     | 90.1% of baseline              |
| ScanNet-V2 (Indoor) | 0.014% (20pts/scene) | 63.8 | 90.1% of baseline |
| SemanticKITTI (Outdoor) | 0.1%       | 53.8     | 85.7% of baseline              |

- Outperforms state-of-the-art active/semi-supervised methods under extremely low annotation budgets.
- Robust to both indoor (diverse scenes) and outdoor (LiDAR scans) point cloud data.

## 🎯 Core Methodology
### 1. Teacher-Student Backbone
- Dual-model architecture (student + teacher) trained on sparse labeled data and abundant unlabeled data.
- Teacher model: Updated via Exponential Moving Average (EMA) of student parameters for stable pseudo-label generation.
- Student model: Trained with supervised loss (labeled data) + unsupervised loss (pseudo-labels) + consistency loss, using sparse convolution for efficient 3D feature extraction.

### 2. Inconsistency-Aware Uncertainty (IAU)
- Combines **hierarchical model uncertainty** (point-level and cluster-level scoring) and **teacher-student prediction inconsistency** (Jensen-Shannon divergence) to evaluate point informativeness.
- Prioritizes points from ambiguous categories (e.g., windows, columns) and rare classes (e.g., sofas) that are critical for performance improvement.

### 3. Uncertainty-Guided Adaptive PseudoLabeling (UGAPL)
- Dynamically weights pseudo-labels using two key signals:
  - Normalized IAU scores (distinguish reliable vs. unreliable pseudo-labels).
  - Real-time guidance ratio (teacher confidence / student confidence) to leverage stable teacher predictions.
- Avoids static pseudo-label filtering, fully utilizing valuable information from all unlabeled data.

## 🔧 Environment Setup
### Quick Installation with Conda
```bash
# Create environment from the provided environment.yaml
conda env create -f environment.yaml

# Activate the TMM environment
conda activate TMM

# Install core dependency (MinkowskiEngine for 3D feature extraction)
pip install MinkowskiEngine==0.5.4

# Compile C++ utility functions for data processing
sh compile_op.sh
```

### Environment Details
| Category       | Key Packages & Versions                                                                 |
|----------------|-----------------------------------------------------------------------------------------|
| Core           | Python 3.6, PyTorch 1.8.0 (CUDA 11.1), TorchVision 0.9.0                                |
| Point Cloud    | Open3D-Python 0.3.0.0, PyNTCLOUD 0.3.1, Plyfile 0.7.4, Trimesh 3.23.5, TorchSparse 1.2.0 |
| Numerical      | NumPy 1.19.2, SciPy 1.5.2, Pandas 1.1.5                                                  |
| ML Tools       | Scikit-Learn 0.24.2, PyTorch-Scatter 2.0.8                                                |
| Others         | CUDA 11.1, Jupyter Notebook 6.4.10, Matplotlib 3.3.4, TQDM 4.61.1                        |

### Dataset Preparation
#### Download Datasets
| Dataset       | Type    | Classes | Download Link                                                                 |
|---------------|---------|---------|--------------------------------------------------------------------------------|
| S3DIS         | Indoor  | 13      | [Google Form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform) |
| ScanNet-V2    | Indoor  | 20      | [Official Website](http://www.scan-net.org/)                                    |
| SemanticKITTI | Outdoor | 19      | [Official Website](http://semantickitti.org/)                                   |

#### Preprocessing Steps
1. **S3DIS**: Extract the dataset and run `data_preparation/data_prepare_s3dis.py` (modify input/output paths in the script).
2. **ScanNet-V2**: Follow the official preprocessing guidelines and use our provided script for format alignment.
3. **SemanticKITTI**: Extract LiDAR scans and annotations, no additional preprocessing required.

#### Processed Dataset Structure
```
data/
├── s3dis/
│   ├── Area_1/
│   │   ├── coords/
│   │   ├── labels/
│   │   ├── rgb/
│   │   └── proj/
│   └── ... (Area_2 to Area_6)
├── scannetv2/
│   ├── scans/
│   ├── scans_test/
│   └── scannetv2-labels.combined.tsv
└── semantickitti/
    └── dataset/
        └── sequences/
            ├── 00/ (training)
            ├── 01/ (training)
            └── ... (up to 11 for testing)
```

## 🚀 Training & Inference
### Step 1: Initialize Labeled Data
```bash
# Generate initial labeled points (random sampling based on annotation budget)
python init_labeled_points.py \
  --dataset s3dis \
  --data_root ./data/s3dis \
  --save_path ./init_labeled \
  --labeled_num_per_scene 50  # Adjust based on total budget
```

### Step 2: Active Semi-Supervised Training
```bash
# Train on S3DIS with 0.02% annotation budget (example)
python train.py \
  --dataset s3dis \
  --data_root ./data/s3dis \
  --init_labeled_path ./init_labeled \
  --annotation_budget 0.02 \
  --active_iterations 2 \
  --batch_size 4 \
  --epochs 60000 \
  --gpu 0  # Single NVIDIA RTX A6000 GPU (compatible with CUDA 11.1)
```

### Step 3: Inference
```bash
# Test the trained model on S3DIS Area-5
python infer.py \
  --dataset s3dis \
  --data_root ./data/s3dis \
  --checkpoint ./checkpoints/best_model.pth \
  --test_area 5 \
  --gpu 0
```

### Code Status
- Core framework code (teacher-student model, IAU, UGAPL) is **in development** and will be released soon.
- Dataset processing scripts and pre-trained checkpoints will be provided upon code release.

## 📈 Results
### Quantitative Results
- On S3DIS (0.02% budget): Outperforms existing methods by 3.1–7.0% mIoU.
- On ScanNet-V2 (20pts/scene): Outperforms SQN by 15.2% mIoU, achieving results comparable to methods using 10× more labels.
- On SemanticKITTI (0.1% budget): Outperforms OneThingOneClick by 28.1% mIoU, setting a new state-of-the-art for ultra-low annotation budgets.

### Qualitative Results
Our method achieves clearer segmentation boundaries for ambiguous categories and better recognition of rare classes:
![Qualitative Comparison](https://example.com/qualitative_results.png)
*(Note: Replace with actual visualization links from the paper)*

## 🧪 Ablation Studies
### Component Effectiveness
| Configuration               | mIoU (%) |
|-----------------------------|----------|
| Baseline (Teacher-Student only) | 54.3 |
| Baseline + IAU              | 58.8     |
| Baseline + UGAPL            | 58.5     |
| Baseline + IAU + UGAPL (Our Method) | 59.0 |

### Pseudo-Label Weight Calculation
| Weight Formula                          | mIoU (%) |
|-----------------------------------------|----------|
| ωₚ = max(pᵗᵢ) - max(pˢᵢ)                | 55.1     |
| ωₚ = max(pᵗᵢ) / max(pˢᵢ)                | 58.6     |
| ωₚ = (1 - U_IE) × max(pᵗᵢ)/max(pˢᵢ)     | 54.7     |
| ωₚ = U_norm × max(pᵗᵢ)/max(pˢᵢ) (Our)   | 59.0     |

## 📝 Citation
If you find this work useful, please cite our paper:
```bibtex
@article{xu2025inconsistency,
  title={Inconsistency-Aware Active Learning for Semi-supervised 3D Point Cloud Semantic Segmentation},
  author={Xu, Zongyi and Jiang, Yu and Zhao, Shanshan and Yuan, Bo and Zhang, Qianni and Fang, Yang and Yang, Xiaoli and Li, Weisheng and Gao, Xinbo},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

## 🎨 Acknowledgements
- Thanks to the open-source communities of PyTorch, MinkowskiEngine, and TorchSparse for their powerful frameworks.
- Dataset processing references official guidelines of S3DIS, ScanNet-V2, and SemanticKITTI.

## 📋 TODO
- [ ] Release full training/inference code
- [ ] Provide pre-trained checkpoints
- [ ] Add detailed dataset processing tutorials
- [ ] Support custom datasets
- [ ] Integrate vision-language foundation models for cold-start problem (future work)

## 📧 Contact
For questions or issues, please contact:
- Zongyi Xu: [xuzy@cqupt.edu.cn](mailto:xuzy@cqupt.edu.cn)
- Xinbo Gao (Corresponding Author): [gaoxb@cqupt.edu.cn](mailto:gaoxb@cqupt.edu.cn)

---

### 🔍 Compatibility Note
All experiments are conducted with PyTorch 1.8.0 + CUDA 11.1. The provided `environment.yaml` ensures full compatibility with no additional configuration required.
