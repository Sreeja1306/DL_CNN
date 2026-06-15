import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data from folder
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
target_names = np.load(os.path.join(DATA_DIR, "target_names.npy"))

# Preprocess
X = X / 16.0  # normalize (digits dataset max pixel = 16)
X = X.reshape(-1, 8, 8, 1).astype(np.float32)

lb = LabelBinarizer()
y_enc = lb.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_enc, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build CNN
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (2, 2), activation="relu", padding="same"),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(10, activation="softmax"),
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1,
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss:     {loss:.4f}")

# Save model and metadata
model.save(os.path.join(MODELS_DIR, "cnn_model.h5"))

meta = {
    "classes": target_names.tolist(),
    "label_binarizer": lb,
    "test_accuracy": float(accuracy),
    "test_loss": float(loss),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
}
with open(os.path.join(MODELS_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nModel and metadata saved to models/")
