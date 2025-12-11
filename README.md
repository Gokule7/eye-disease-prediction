# Eye Disease Classification with Grad-CAM Visualization

A deep learning project for automated eye disease classification using fundus images with Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.

## 🎯 Overview

This project implements a Convolutional Neural Network (CNN) for classifying eye diseases from retinal fundus images using the ODIR-5K dataset. It includes Grad-CAM visualization to provide interpretable insights into model predictions, making it suitable for clinical applications.

## 🏥 Disease Categories

The system classifies the following conditions:
- **N**: Normal fundus
- **D**: Diabetic Retinopathy
- **G**: Glaucoma
- **C**: Cataract
- **A**: Age-Related Macular Degeneration (ARMD)
- **H**: Hypertensive Retinopathy
- **M**: Myopia
- **O**: Other conditions

## ✨ Features

- **Deep Learning Classification**: CNN-based eye disease detection
- **Grad-CAM Visualization**: Visual explanations showing where the model focuses
- **Clinical Interpretability**: 3-panel visualizations (original, heatmap, overlay)
- **Batch Processing**: Process multiple images efficiently
- **High-Quality Output**: 300 DPI publication-ready figures
- **Composite Figure Generation**: Combine multiple visualizations

## 📁 Project Structure

```
EyeProject/
├── gradcam_visualization.py      # Core Grad-CAM implementation
├── run_gradcam.py                # Main script for trained models
├── simple_gradcam_demo.py        # Working demo (no training needed)
├── create_composite_figure.py    # Generate composite figures
├── notebookff41799b39.ipynb      # Training notebook
├── full_df.csv                   # Dataset metadata
├── GRADCAM_README.md             # Detailed documentation
├── gradcam_results/              # Generated visualizations
│   ├── gradcam_1_1000_left.jpg
│   ├── gradcam_2_1000_right.jpg
│   ├── gradcam_3_1001_left.jpg
│   ├── gradcam_4_1001_right.jpg
│   └── gradcam_5_1002_left.jpg
├── Figure_8_GradCAM_Composite.png  # Composite figure
└── ODIR-5K/                      # Dataset (not included in repo)
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
TensorFlow 2.13.0
OpenCV (cv2)
NumPy 1.24.3
Matplotlib
Pandas
Pillow
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eye-disease-gradcam.git
cd eye-disease-gradcam
```

2. Install dependencies:
```bash
pip install tensorflow==2.13.0 opencv-python numpy==1.24.3 matplotlib pandas pillow
```

3. Download the ODIR-5K dataset (not included):
   - Place in `ODIR-5K/ODIR-5K/` directory
   - Dataset available at: [ODIR-5K Dataset](https://odir2019.grand-challenge.org/)

### Run Demo (No Training Required)

```bash
python simple_gradcam_demo.py
```

This generates 5 Grad-CAM visualizations in the `gradcam_results/` folder.

### Create Composite Figure

```bash
python create_composite_figure.py
```

Generates `Figure_8_GradCAM_Composite.png` with all visualizations stacked vertically.

## 📊 Using with Your Trained Model

1. **Train your model** using `notebookff41799b39.ipynb`

2. **Save the model**:
```python
model.save('my_eye_disease_model.keras')
```

3. **Update `run_gradcam.py`** with your model path:
```python
MODEL_PATH = "path/to/your/model.keras"
```

4. **Run Grad-CAM**:
```bash
python run_gradcam.py
```

## 🎨 Grad-CAM Visualization

Each visualization includes three panels:
- **Left**: Original fundus image
- **Center**: Grad-CAM attention heatmap (jet colormap)
- **Right**: Overlay showing model attention + prediction

### Example Output

The Grad-CAM heatmaps highlight:
- Regions the model focuses on for predictions
- Clinical relevance of attention areas
- Model interpretability for medical professionals

## 📖 How It Works

### Grad-CAM Algorithm

1. Forward pass through the model
2. Compute gradients of predicted class w.r.t. last conv layer
3. Weight feature maps by gradient importance
4. Generate heatmap showing influential regions
5. Overlay heatmap on original image

### Key Components

- **GradCAM Class**: Core implementation with automatic layer detection
- **Heatmap Generation**: Gradient-weighted activation mapping
- **Visualization**: High-quality medical imaging outputs

## 🏥 Clinical Relevance

Grad-CAM visualizations provide:
- ✅ **Interpretability**: See what the model "sees"
- ✅ **Validation**: Verify clinically relevant features
- ✅ **Trust**: Transparent AI decision-making
- ✅ **Error Detection**: Identify focus on artifacts vs pathology
- ✅ **Education**: Training tool for medical professionals

## 📈 Results

- **5 Demonstration Images**: Successfully processed
- **Success Rate**: 100% (5/5)
- **Output Quality**: 300 DPI, publication-ready
- **Composite Figure**: 4432×7670 pixels

## 🔬 Technical Details

- **Input Size**: 224×224 pixels (configurable)
- **Heatmap Method**: Gradient-weighted class activation
- **Colormap**: Jet (medical imaging standard)
- **Overlay Alpha**: 0.4 (40% heatmap, 60% original)
- **Architecture Support**: CNN, ResNet, EfficientNet, MobileNet

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{eye_disease_gradcam,
  title={Eye Disease Classification with Grad-CAM Visualization},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/eye-disease-gradcam}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or collaboration:
- GitHub Issues: [Project Issues](https://github.com/yourusername/eye-disease-gradcam/issues)
- Email: gokule710@gmail.com

## ⚠️ Disclaimer

This tool is for research and educational purposes only. Not intended for clinical diagnosis without proper validation and regulatory approval.

---

**Status**: ✅ Operational  
**Last Updated**: November 29, 2025  
**Version**: 1.0.0
