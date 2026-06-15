# CNN Classification - Sklearn Digits

Convolutional Neural Network for handwritten digit recognition using the Sklearn Digits dataset.

## Project Structure

```
CNN_Classification/
    data/               # Dataset files (.npy) loaded locally
    models/             # Trained model (.h5) and metadata (.pkl)
    notebooks/          # Jupyter notebook
    static/             # CSS stylesheet
    app.py              # Streamlit frontend
    train_model.py      # Training script
    requirements.txt
    README.md
```

## Dataset

- **Source:** Sklearn built-in Digits dataset (loaded from local `data/` folder)
- **Samples:** 1,797 images
- **Image Size:** 8x8 grayscale pixels
- **Classes:** 10 (digits 0-9)
- **Split:** 70% train / 15% validation / 15% test

## CNN Architecture

| Layer        | Type         | Output Shape  |
|--------------|--------------|---------------|
| conv2d       | Conv2D       | (None,8,8,32) |
| max_pooling  | MaxPooling2D | (None,4,4,32) |
| dropout_1    | Dropout(0.25)| -             |
| conv2d_1     | Conv2D       | (None,4,4,64) |
| flatten      | Flatten      | (None,1024)   |
| dense        | Dense(128)   | (None,128)    |
| dropout_2    | Dropout(0.4) | -             |
| dense_1      | Dense(10)    | (None,10)     |

**Optimizer:** Adam (lr=0.001) | **Loss:** Categorical Crossentropy | **Epochs:** 30

## Steps to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (saves to `models/`):
   ```
   python train_model.py
   ```

3. Launch the app:
   ```
   streamlit run app.py
   ```

## App Tabs

- **Model Summary** - Architecture table and compile settings
- **Evaluate** - Test accuracy, loss, classification report, confusion matrix
- **Predict** - Single sample prediction with pixel visualization and class probabilities
- **About Dataset** - Dataset details and split information
