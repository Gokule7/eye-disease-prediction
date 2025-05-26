# 🧿 Eye Disease Prediction System

A Deep Learning–Based Intelligent Ocular Disease Prediction System built with Flutter (UI), Flask (API backend), and EfficientNetB3 (AI model). It can classify fundus images into 8 eye disease categories.

## 🩺 Supported Diseases

- N: Normal
- D: Diabetes
- G: Glaucoma
- C: Cataract
- A: Age-related Macular Degeneration
- H: Hypertension
- M: Pathological Myopia
- O: Other diseases/abnormalities

---

## 🚀 Features

- Upload **Left Eye** and **Right Eye** images.
- Real-time prediction using trained EfficientNetB3 model.
- Displays result on the mobile device.
- Sends prediction result as PDF to registered email.
- Responsive Flutter UI for both web and mobile.
- Flask backend with multi-class model support.

---

## 📁 Project Structure

eye_disease_app/
├── backend/
│ ├── app.py # Flask backend
│ ├── model/ # Trained EfficientNetB3 model files
│ ├── utils/ # Helper functions
│ └── requirements.txt
├── frontend/
│ ├── lib/ # Flutter app code
│ └── pubspec.yaml
├── README.md
└── LICENSE

---

## 🧠 Model Details

- **Model Architecture**: EfficientNetB3
- **Framework**: TensorFlow + Keras
- **Dataset**: ODIR (Ocular Disease Intelligent Recognition) dataset
- **Accuracy Achieved**: ~96.19%
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score

---

## ⚙️ How to Run

### 🔧 Backend (Flask)

```bash
cd backend
pip install -r requirements.txt
flask --app app run

📱 Frontend (Flutter)

cd frontend
flutter pub get
flutter run

📦 Dependencies

Flutter SDK

Flask

TensorFlow / Keras

OpenCV

NumPy / PIL / sklearn