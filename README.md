# Inconsistency-Aware Active Learning for Semi-supervised 3D Point Cloud Semantic Segmentation
[![IEEE TRANSACTIONS ON MULTIMEDIA](https://img.shields.io/badge/Paper-IEEE%20TMM%202025-blue)](https://ieeexplore.ieee.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%201.8.0-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.6-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🌟 Overview
This repository implements an **inconsistency-aware active semi-supervised learning framework** for 3D point cloud semantic segmentation. The core goal is to achieve near-fully-supervised performance with extremely limited labeled data (as low as 0.014% of training data), by integrating active learning (AL) and semi-supervised learning (SSL) in a synergistic manner.

### Key Contributions
1. **Inconsistency-Aware Uncertainty (IAU)**: Enhances active learning sample selection by leveraging prediction inconsistencies between teacher-student models, prioritizing hard-to-classify points.
2. **Uncertainty-Guided Adaptive PseudoLabeling (UGAPL)**: Dynamically adjusts pseudo-label weights using AL-derived uncertainty and real-time model confidence, mitigating noise in pseudo-labels.
3. **Deep AL-SSL Integration**: Achieves state-of-the-art performance on indoor (S3DIS, ScanNet) and outdoor (SemanticKITTI) datasets with minimal annotations.

## 📊 Performance
| Dataset       | Annotation Budget | mIoU (%) | Fully-Supervised Baseline (%) |
|---------------|-------------------|----------|--------------------------------|
| S3DIS (Indoor) | 0.02%             | 59.0     | 90.1% of baseline              |
| ScanNet-V2 (Indoor) | 0.014% (20pts/scene) | 63.8 | 90.1% of baseline |
| SemanticKITTI (Outdoor) | 0.1%       | 53.8     | 85.7% of baseline              |

- Outperforms existing active/semi-supervised methods (e.g., HPAL, SQN, HybridCR) under extremely low annotation budgets.
- Robust to both indoor (diverse scenes) and outdoor (LiDAR scans) point cloud data.

## 🎯 Core Methodology
The framework consists of three key components:
### 1. Teacher-Student Backbone
- Dual-model architecture (student + teacher) trained on sparse labeled data and abundant unlabeled data.
- Teacher model: Updated via Exponential Moving Average (EMA) of student parameters for stable pseudo-label generation.
- Student model: Trained with supervised loss (labeled data) + unsupervised loss (pseudo-labels) + consistency loss.

### 2. Inconsistency-Aware Uncertainty (IAU)
- Combines **model uncertainty** (hierarchical uncertainty from HPAL) and **teacher-student prediction inconsistency** (Jensen-Shannon divergence) to score point informativeness.
- Prioritizes points from ambiguous/rare categories (e.g., windows, columns, sofas) that are critical for performance improvement.

### 3. Uncertainty-Guided Adaptive PseudoLabeling (UGAPL)
- Dynamically weights pseudo-labels using:
  - Normalized IAU scores (distinguishes reliable/unreliable pseudo-labels).
  - Real-time guidance ratio (teacher confidence / student confidence).
- Avoids pseudo-label filtering (common in SSL) to fully utilize information from unlabeled data.

## 🔧 Environment Setup
### Quick Installation with Conda
The environment can be directly reconstructed using the provided `environment.yaml` file:
```bash
# Create environment from yaml file
conda env create -f environment.yaml

# Activate the TMM environment
conda activate TMM

# Install missing required package (MinkowskiEngine 0.5.4)
pip install MinkowskiEngine==0.5.4
```

### Environment Details
| Category       | Key Packages & Versions                                                                 |
|----------------|-----------------------------------------------------------------------------------------|
| Core           | Python 3.6, PyTorch 1.8.0 (CUDA 11.1), TorchVision 0.9.0                                |
| Point Cloud    | Open3D-Python 0.3.0.0, PyNTCLOUD 0.3.1, Plyfile 0.7.4, Trimesh 3.23.5, TorchSparse 1.2.0 |
| Numerical      | NumPy 1.19.2, SciPy 1.5.2, Pandas 1.1.5                                                  |
| ML Tools       | Scikit-Learn 0.24.2, PyTorch-Scatter 2.0.8                                                |
| Others         | CUDA 11.1, Jupyter Notebook 6.4.10, Matplotlib 3.3.4, TQDM 4.61.1                        |

### Environment YAML File
The complete `environment.yaml` is provided in the repository. It includes all conda and pip dependencies for one-click installation.

### Dataset Preparation
We evaluate on 3 large-scale point cloud datasets. Download and organize as follows:
| Dataset       | Type    | Classes | Download Link                                                                 |
|---------------|---------|---------|--------------------------------------------------------------------------------|
| S3DIS         | Indoor  | 13      | [Official Website](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform) |
| ScanNet-V2    | Indoor  | 20      | [Official Website](http://www.scan-net.org/)                                    |
| SemanticKITTI | Outdoor | 19      | [Official Website](http://semantickitti.org/)                                   |

#### Dataset Organization Structure
```
data/
├── s3dis/
│   ├── Area_1/
│   ├── Area_2/
│   └── ... (Area_3 to Area_6)
├── scannetv2/
│   ├── scans/
│   ├── scans_test/
│   └── scannetv2-labels.combined.tsv
└── semantickitti/
    ├── dataset/
    │   ├── sequences/
    │   │   ├── 00/ (training)
    │   │   ├── 01/ (training)
    │   │   └── ... (up to 11 for testing)
```

## 🚀 Training & Inference
### Training Pipeline
1. **Initialize**: Randomly sample `B/Iter` points (B=annotation budget, Iter=active iterations) as initial labeled data.
2. **SSL Training**: Train teacher-student model on labeled + unlabeled data.
3. **Active Selection**: Use IAU to select high-informativeness points, send to oracle for labeling.
4. **Update Labeled Set**: Add newly labeled points to the training set.
5. **UGAPL Training**: Retrain the model with dynamically weighted pseudo-labels.
6. **Repeat**: Iterate until annotation budget is exhausted.

### Training Command
```bash
# Activate environment
conda activate TMM

# Example: Train on S3DIS with 0.02% annotation budget
python train.py \
  --dataset s3dis \
  --data_root ./data/s3dis \
  --annotation_budget 0.02 \
  --active_iterations 2 \
  --batch_size 4 \
  --epochs 60000 \
  --gpu 0  # Use single NVIDIA RTX A6000 GPU (compatible with CUDA 11.1)
```

### Inference Command
```bash
# Activate environment
conda activate TMM

# Example: Inference on S3DIS Area-5
python infer.py \
  --dataset s3dis \
  --data_root ./data/s3dis \
  --checkpoint ./checkpoints/best_model.pth \
  --split test \
  --gpu 0
```

### Code Status
- Core framework code (teacher-student model, IAU, UGAPL) is **in development** and will be released soon.
- Dataset processing scripts for S3DIS/ScanNet/SemanticKITTI are **in development**.
- Pre-trained checkpoints will be provided after code release.

## 📈 Results
### Quantitative Results
- On S3DIS (0.02% budget): Outperforms HPAL by 3.1% mIoU, ERDA by 7.0% mIoU.
- On ScanNet-V2 (20pts/scene): Outperforms SQN by 15.2% mIoU, HPAL by 1.3% mIoU.
- On SemanticKITTI (0.1% budget): Outperforms OneThingOneClick by 28.1% mIoU.

### Qualitative Results
Our method achieves clearer segmentation boundaries for ambiguous categories (e.g., windows, columns) and better recognition of rare classes (e.g., sofas) compared to HPAL:
![Qualitative Comparison](https://example.com/qualitative_results.png)
*(Note: Replace with actual visualization links from the paper)*

## 🧪 Ablation Studies
### Component Effectiveness
| Configuration       | mIoU (%) |
|---------------------|----------|
| Baseline (Teacher-Student only) | 54.3 |
| Baseline + IAU      | 58.8     |
| Baseline + UGAPL    | 58.5     |
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
- Backbone code is based on [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
- Dataset processing references [HPAL](https://github.com/zongyi-xu/HPAL) and [SemanticKITTI API](https://github.com/PRBonn/semantic-kitti-api).

## 📋 TODO
- [ ] Release full training/inference code
- [ ] Provide pre-trained checkpoints
- [ ] Add dataset processing scripts (compatible with current environment)
- [ ] Support custom datasets
- [ ] Integrate vision-language foundation models for cold-start problem (future work)

## 📧 Contact
For questions or issues, please contact:
- Zongyi Xu: [xuzy@cqupt.edu.cn](mailto:xuzy@cqupt.edu.cn)
- Xinbo Gao (Corresponding Author): [gaoxb@cqupt.edu.cn](mailto:gaoxb@cqupt.edu.cn)

---

### 🔍 Environment Compatibility Note
All experiments in the paper were conducted with PyTorch 1.8.0 + CUDA 11.1, which matches the `TMM` conda environment defined in `environment.yaml`. No additional configuration is required except installing MinkowskiEngine 0.5.4 after environment creation.
