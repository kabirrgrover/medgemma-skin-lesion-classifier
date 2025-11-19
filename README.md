# MedGemma Skin Lesion Classifier

A production-minded MVP for classifying 7 common skin lesion types using Google's MedGemma-4B vision encoder, with a focus on melanoma detection for medical screening applications.

## 🎯 Project Goal

Develop a reliable skin lesion classification system that:
- **Prioritizes melanoma detection** (high sensitivity for early screening)
- Provides **calibrated confidence** estimates for clinical decision support
- Offers **visual explainability** (Grad-CAM) to help clinicians understand predictions
- Serves as a foundation for a production medical imaging pipeline

## 📊 Dataset

**HAM10000** - Human Against Machine with 10,000 training images
- 7 lesion types: Actinic keratoses (akiec), Basal cell carcinoma (bcc), Benign keratosis (bkl), Dermatofibroma (df), **Melanoma (mel)**, Melanocytic nevi (nv), Vascular lesions (vasc)
- Significant class imbalance (melanoma is rare but critical)
- Stratified train/val/test splits (70/15/15) to maintain class distribution

## 🧠 Model Architecture

**MedGemma-4B** (Google's multimodal medical foundation model)
- **Vision Encoder**: SigLIP-based transformer (frozen during training)
- **Input**: 896×896 RGB images (MedGemma's native resolution)
- **Classifier Head**: Lightweight trainable layer on top of frozen encoder
- **Design**: Supports future QLoRA fine-tuning for full model adaptation

### Training Approach

1. **Classifier-Head Fine-Tuning**: Train only the classification layer while keeping the pre-trained vision encoder frozen
   - Efficient training with minimal parameters
   - Preserves rich medical image representations from MedGemma

2. **Class Imbalance Handling**:
   - **Class Weighting**: Inverse frequency weighting with melanoma boost
   - **Focal Loss**: Focuses learning on hard, misclassified examples
   - **Hybrid Early Stopping**: Balances melanoma recall and overall macro F1

3. **Mixed Precision Training**:
   - FP16 for encoder (memory efficient)
   - FP32 for classifier (numerical stability)

## 🔬 Key Features

### ✅ Implemented

- **Stratified Data Splitting**: Maintains class distribution across splits
- **Class-Weighted Training**: Addresses severe class imbalance
- **Focal Loss**: Handles hard examples and class imbalance
- **Hybrid Early Stopping**: Multi-metric optimization (melanoma recall + macro F1)
- **Comprehensive Evaluation**: Per-class metrics, confusion matrices, melanoma-specific analysis

### 🚧 Planned

- **Grad-CAM Visualizations**: Heatmaps showing which image regions drive predictions
- **Monte Carlo Dropout**: Uncertainty quantification for confidence calibration
- **Gradio Demo**: Interactive web interface for testing
- **FastAPI Endpoint**: REST API for integration

## 📁 Project Structure

```
medproj/
├── src/
│   ├── models/
│   │   └── medgemma_wrapper.py    # MedGemma model wrapper
│   ├── training/
│   │   ├── dataset.py             # HAM10000 dataset loader
│   │   └── trainer.py             # Training utilities
│   └── utils/
│       └── helpers.py             # Class mappings, weights
├── scripts/
│   ├── preprocess_data.py        # Data preprocessing & splitting
│   ├── train.py                  # Main training script
│   └── evaluate.py               # Model evaluation
├── configs/                      # Hydra configuration files
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   └── training/
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- HuggingFace account with access to [MedGemma-4B](https://huggingface.co/google/medgemma-4b-it) (gated model)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd medproj

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace
huggingface-cli login
```

### Data Setup

1. Download HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Place in `data/raw/` directory
3. Run preprocessing:

```bash
python scripts/preprocess_data.py \
  --input data/raw/ \
  --output data/processed/ \
  --image-size 896 \
  --splits 0.7 0.15 0.15
```

### Training

```bash
python scripts/train.py \
  model=medgemma \
  data=ham10000 \
  training=default
```

### Evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/classifier_weights.pth \
  --split test
```

## 🛠️ Methods & Techniques

### Class Imbalance Strategies

- **Inverse Frequency Weighting**: Standard approach to balance class contributions
- **Square Root Weighting**: Moderate weighting to prevent overcompensation
- **Melanoma Boost**: Additional weight multiplier for critical class
- **Focal Loss**: `FL(p_t) = -α(1-p_t)^γ log(p_t)` - down-weights easy examples

### Training Techniques

- **Gradient Accumulation**: Simulates larger batch sizes with limited GPU memory
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: FP16/FP32 hybrid for efficiency and stability

### Evaluation Metrics

- **Melanoma Recall (Sensitivity)**: Primary metric - percentage of melanomas detected
- **Melanoma Precision**: Percentage of melanoma predictions that are correct
- **Macro F1**: Average F1 across all classes (handles imbalance)
- **Weighted F1**: F1 weighted by class frequency
- **Confusion Matrix**: Detailed per-class performance breakdown

## 📚 Technical Details

### Model Specifications

- **Base Model**: `google/medgemma-4b-it`
- **Vision Encoder**: SigLIP Vision Transformer
- **Encoder Output Dim**: 1152
- **Image Size**: 896×896 (MedGemma's native resolution)
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training Configuration

- **Optimizer**: AdamW
- **Learning Rate**: 3e-5 to 5e-5 (with scheduling)
- **Batch Size**: 4-8 (depending on GPU memory)
- **Epochs**: 20-30 (with early stopping)
- **Dropout**: 0.1
- **Training Hardware**: NVIDIA A100-SXM4-40GB GPU (Google Colab Pro)
- **Training Time**: ~3 hours for 30 epochs on A100

## 🔍 Explainability & Uncertainty

### Grad-CAM (Planned)

Gradient-weighted Class Activation Mapping will generate heatmaps showing:
- Which image regions the model focuses on
- Visual validation that the model attends to lesion areas
- Interpretability for clinical review

### Monte Carlo Dropout (Planned)

Uncertainty quantification through:
- Multiple forward passes with dropout enabled
- Confidence intervals for predictions
- Calibrated probability estimates

## ⚠️ Important Notes

- **Not for Clinical Use**: This is a research/educational project. Do not use for actual medical diagnosis.
- **Model Access**: MedGemma-4B requires HuggingFace access approval
- **Data License**: HAM10000 dataset has its own usage terms
- **Bias Considerations**: Model performance varies by lesion type; melanoma detection is prioritized

## 📖 References

- **MedGemma**: [Google's Medical Multimodal Foundation Model](https://huggingface.co/google/medgemma-4b-it)
- **HAM10000**: [Skin Cancer MNIST Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)

## 🚀 Next Steps & Future Work

### Immediate Priorities

1. **Gradio Web Interface**
   - Interactive demo for image upload and prediction
   - Real-time Grad-CAM visualization
   - Confidence scores and uncertainty display
   - User-friendly interface for clinicians and researchers

2. **FastAPI REST Endpoint**
   - Production-ready API for integration
   - Batch prediction support
   - Model versioning and health checks
   - Documentation with OpenAPI/Swagger

3. **Enhanced Calibration**
   - Implement Platt scaling and isotonic regression
   - Compare calibration methods on validation set
   - Create calibration curves and reliability diagrams

### Research & Development

4. **External Validation**
   - Test on additional datasets (ISIC, PH2)
   - Cross-dataset generalization analysis
   - Fairness evaluation across demographic groups

5. **Model Improvements**
   - Experiment with deeper classifier architectures
   - Explore ensemble methods for robustness
   - Investigate QLoRA fine-tuning of encoder
   - Advanced data augmentation (mixup, cutmix)

6. **Production Deployment**
   - Model quantization for edge devices
   - ONNX export for cross-platform inference
   - Docker containerization
   - CI/CD pipeline for model updates

### Long-Term Vision

7. **Clinical Integration**
   - Prospective validation studies
   - Integration with PACS systems
   - Real-world performance monitoring
   - Feedback loop for continuous improvement

8. **Advanced Features**
   - Multi-task learning (segmentation + classification)
   - Temporal analysis for lesion tracking
   - Integration with patient history
   - Explainability for multi-class predictions

## 🤝 Contributing

This is a research project. Contributions, issues, and discussions are welcome!

## 📄 License

See LICENSE file for details. Note that:
- MedGemma model has its own license terms
- HAM10000 dataset has its own usage terms
- Project code is provided as-is for educational purposes

---

**Disclaimer**: This project is for research and educational purposes only. It is not intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

