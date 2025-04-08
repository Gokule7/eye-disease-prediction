from PIL import Image
import numpy as np
import tensorflow as tf

LABELS = ["Normal (N)", "Diabetes (D)", "Glaucoma (G)", "Cataract (C)", "Age-related Macular Degeneration (A)", "Hypertension (H)", "Pathological Myopia (M)", "Other (O)"]

def preprocess(image, size=(224, 224)):
    img = Image.open(image).convert('RGB')
    img = img.resize(size)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict_disease(left_img, right_img, model):
    left = preprocess(left_img)
    right = preprocess(right_img)
    
    # Combine predictions
    left_pred = model.predict(left)[0]
    right_pred = model.predict(right)[0]
    
    avg_pred = (left_pred + right_pred) / 2.0
    top_index = np.argmax(avg_pred)
    
    result = {
        "top_label": LABELS[top_index],
        "probabilities": {LABELS[i]: round(float(p) * 100, 2) for i, p in enumerate(avg_pred)}
    }
    return result
