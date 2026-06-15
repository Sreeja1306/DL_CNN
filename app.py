import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSS_PATH = os.path.join(BASE_DIR, "static", "style.css")

st.set_page_config(page_title="CNN Digit Classifier", layout="wide")

# Load CSS
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_meta():
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_model.h5"))
    with open(os.path.join(MODELS_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return model, meta


@st.cache_data
def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    X_norm = X / 16.0
    X_reshaped = X_norm.reshape(-1, 8, 8, 1).astype(np.float32)
    lb = LabelBinarizer()
    y_enc = lb.fit_transform(y)
    _, X_temp, _, y_temp = train_test_split(X_reshaped, y_enc, test_size=0.3, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_test, y_test, lb


# Header
st.markdown('<div class="header-block"><h1>CNN Digit Classifier</h1><p>Convolutional Neural Network trained on the Sklearn Digits Dataset</p></div>', unsafe_allow_html=True)

# Load
model, meta = load_model_and_meta()
X_test, y_test, lb = load_data()

# Sidebar
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(f"**Dataset:** Sklearn Digits")
st.sidebar.markdown(f"**Classes:** {', '.join(str(c) for c in meta['classes'])}")
st.sidebar.markdown(f"**Train Samples:** {meta['train_samples']}")
st.sidebar.markdown(f"**Test Samples:** {meta['test_samples']}")
st.sidebar.markdown(f"**Test Accuracy:** {meta['test_accuracy']*100:.2f}%")
st.sidebar.markdown(f"**Test Loss:** {meta['test_loss']:.4f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Model Summary", "Evaluate", "Predict", "About Dataset"])

# --- TAB 1: Model Summary (Compile Info) ---
with tab1:
    st.markdown("### Architecture")
    rows = []
    for layer in model.layers:
        cfg = layer.get_config()
        ltype = layer.__class__.__name__
        params = layer.count_params()
        shape = str(layer.output.shape) if hasattr(layer, 'output') else "-"
        rows.append({"Layer": layer.name, "Type": ltype, "Output Shape": shape, "Parameters": params})

    st.table(rows)

    st.markdown("### Compile Settings")
    col1, col2, col3 = st.columns(3)
    col1.metric("Optimizer", "Adam")
    col2.metric("Learning Rate", "0.001")
    col3.metric("Loss Function", "Categorical Crossentropy")

    col4, col5 = st.columns(2)
    col4.metric("Epochs", "30")
    col5.metric("Batch Size", "32")

# --- TAB 2: Evaluate ---
with tab2:
    st.markdown("### Test Set Evaluation")

    col1, col2 = st.columns(2)
    col1.metric("Test Accuracy", f"{meta['test_accuracy']*100:.2f}%")
    col2.metric("Test Loss", f"{meta['test_loss']:.4f}")

    st.markdown("---")

    if st.button("Run Full Evaluation"):
        with st.spinner("Evaluating on test set..."):
            y_pred_prob = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_test, axis=1)

            report = classification_report(y_true, y_pred, target_names=[str(c) for c in meta['classes']], output_dict=True)
            cm = confusion_matrix(y_true, y_pred)

        st.markdown("#### Classification Report")
        report_rows = []
        for label in [str(c) for c in meta['classes']]:
            report_rows.append({
                "Class": label,
                "Precision": f"{report[label]['precision']:.2f}",
                "Recall": f"{report[label]['recall']:.2f}",
                "F1-Score": f"{report[label]['f1-score']:.2f}",
                "Support": int(report[label]['support'])
            })
        st.table(report_rows)

        st.markdown("#### Confusion Matrix")
        cm_display = []
        for i, row in enumerate(cm):
            cm_display.append({"True \\ Pred": str(i), **{str(j): int(v) for j, v in enumerate(row)}})
        st.table(cm_display)

# --- TAB 3: Predict ---
with tab3:
    st.markdown("### Single Sample Prediction")
    st.markdown("Select a sample index from the test set to run a prediction.")

    sample_idx = st.slider("Test Sample Index", min_value=0, max_value=len(X_test) - 1, value=0)

    sample = X_test[sample_idx]
    true_label = int(np.argmax(y_test[sample_idx]))

    col1, col2 = st.columns([1, 2])

    with col1:
        pixel_grid = (sample.reshape(8, 8) * 255).astype(int)
        pixel_html = '<div class="pixel-grid">'
        for row in pixel_grid:
            for val in row:
                darkness = int(val)
                color = f"rgb({255-darkness},{255-darkness},{255-darkness})"
                pixel_html += f'<div class="pixel" style="background:{color};"></div>'
        pixel_html += "</div>"
        st.markdown(pixel_html, unsafe_allow_html=True)
        st.markdown(f"**True Label:** {true_label}")

    with col2:
        pred_prob = model.predict(sample.reshape(1, 8, 8, 1), verbose=0)[0]
        predicted = int(np.argmax(pred_prob))
        confidence = float(pred_prob[predicted]) * 100

        status = "Correct" if predicted == true_label else "Incorrect"
        status_color = "green" if predicted == true_label else "red"

        st.markdown(f"**Predicted Label:** {predicted}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.markdown(f'**Status:** <span style="color:{status_color};font-weight:bold;">{status}</span>', unsafe_allow_html=True)

        st.markdown("#### Class Probabilities")
        prob_rows = [{"Digit": str(i), "Probability": f"{p*100:.2f}%"} for i, p in enumerate(pred_prob)]
        st.table(prob_rows)

# --- TAB 4: About Dataset ---
with tab4:
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", "1,797")
    col2.metric("Image Size", "8 x 8 pixels")
    col3.metric("Classes", "10 (digits 0-9)")

    st.markdown("""
The **Sklearn Digits** dataset is a built-in dataset from `sklearn.datasets`.
Each sample is an 8x8 grayscale image of a handwritten digit (0-9), represented as 64 pixel values ranging from 0 to 16.

The data is loaded directly from the local `data/` folder (saved as `.npy` files), not from any URL.

**Split used:**
- 70% Training
- 15% Validation
- 15% Testing
    """)
