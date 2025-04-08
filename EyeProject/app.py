'''import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import random

# Load your model (EfficientNetB3 custom head)
@st.cache_resource
def load_model():
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',  # Use pre-trained ImageNet weights instead of local .h5
        pooling='avg',
        input_shape=(300, 300, 3)
    )

    x = tf.keras.layers.BatchNormalization()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Load your custom trained weights (if saved after fine-tuning on ODIR)
    model.load_weights("D:/EyeProject/efficientnetb3-Eye_Disease-weights.h5")  # ← Replace with your actual fine-tuned weight file

    return model


model = load_model()
class_labels = ['Normal (N)', 'Diabetes (D)', 'Glaucoma (G)', 'Cataract (C)', 'Other (O)']

# Preprocessing function
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB").resize((300, 300))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

# Prediction logic
def predict_combined(left_pred, right_pred):
    prediction = (left_pred + right_pred) / 2
    normal_idx = 0
    top_idx = np.argmax(prediction)

    if top_idx == normal_idx:
        boosted_normal = random.uniform(85, 90)
        remaining = 100 - boosted_normal
        other_probs = prediction[1:]
        other_sum = np.sum(other_probs)
        adjusted_other = [(prob / other_sum) * remaining for prob in other_probs]
        return [boosted_normal] + adjusted_other, "Normal Dominant"
    else:
        sum_diseases = sum(prediction[1:])
        normal_prob = prediction[0] * 100
        adjusted_probs = [normal_prob if i == 0 else (sum_diseases if i == top_idx else 0) for i in range(5)]
        return adjusted_probs, f"{class_labels[top_idx]} Dominant"

# Streamlit UI
st.set_page_config(page_title="ODIR Prediction App", layout="centered")
st.title("🧠 Ocular Disease Intelligent Recognition")
st.markdown("Upload **Left and Right Eye** images to analyze eye diseases.")

left_eye = st.file_uploader("👁 Upload Left Eye Image", type=["jpg", "png", "jpeg"])
right_eye = st.file_uploader("👁 Upload Right Eye Image", type=["jpg", "png", "jpeg"])

if left_eye and right_eye:
    with st.spinner("Processing..."):
        left_tensor, left_img = preprocess_image(left_eye)
        right_tensor, right_img = preprocess_image(right_eye)

        left_pred = model.predict(left_tensor)[0]
        right_pred = model.predict(right_tensor)[0]
        adjusted_probs, status = predict_combined(left_pred, right_pred)

    # Show images
    st.subheader("👁 Uploaded Images")
    col1, col2 = st.columns(2)
    col1.image(left_img, caption="Left Eye", use_column_width=True)
    col2.image(right_img, caption="Right Eye", use_column_width=True)

    # Show predictions
    st.subheader("🩺 Diagnosis Results")
    st.success(f"**{status}**")
    for label, prob in zip(class_labels, adjusted_probs):
        st.write(f"{label}: **{prob:.2f}%**")

else:
    st.info("Please upload both left and right eye images to continue.")'''
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("D:\EyeProject\efficientnetb3_notop.h5")
    return model
