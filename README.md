# Skin Disease Classification Using Deep Learning Techniques

## Project Overview

This project implements deep learning techniques to classify 22 different types of skin diseases from images. It compares three different approaches:

1. Custom CNN model
2. Unsupervised feature extraction with autoencoder + classifier
3. Transfer learning with DenseNet121

## Dataset

The dataset contains images of 22 distinct skin disease classes:

- Acne
- Actinic Keratosis
- Benign Tumors
- Bullous
- Candidiasis
- Drug Eruption
- Eczema
- Infestations/Bites
- Lichen
- Lupus
- Moles
- Psoriasis
- Rosacea
- Seborrheic Keratoses
- Skin Cancer
- Sun/Sunlight Damage
- Tinea
- Unknown/Normal
- Vascular Tumors
- Vasculitis
- Vitiligo
- Warts

## Project Structure

The notebook implements the following steps:

1. Data Preparation and Exploration
2. Data Preprocessing
3. Custom CNN Implementation
4. Unsupervised Learning with Autoencoder for Feature Extraction
5. State-of-the-Art Model Implementation using DenseNet121
6. Model Testing and Performance Comparison

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL (Pillow)

## Setup Instructions

1. Clone the repository:

   ```
   git clone <repository-url>
   cd Skin-Disease
   ```

2. Install the required dependencies:

   ```
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow
   ```

3. Download the dataset and organize it into the following structure:
   ```
   data/
   ├── original data/
   │   ├── train_original_data/
   │   │   ├── Acne/
   │   │   ├── Actinic Keratosis/
   │   │   └── ...
   │   ├── val_original_data/
   │   │   ├── Acne/
   │   │   ├── Actinic Keratosis/
   │   │   └── ...
   │   └── test_original_data/
   │       ├── Acne/
   │       ├── Actinic Keratosis/
   │       └── ...
   ```

## Running the Notebook

1. Launch Jupyter Notebook:

   ```
   jupyter notebook
   ```

2. Open `SkinDisease_Project.ipynb` in your browser.

3. If using Google Colab:

   - Upload the notebook to Google Drive
   - Update the file paths to match your Google Drive structure
   - Make sure to mount your Google Drive using:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```

4. Run all cells in the notebook sequentially.

## Model Architectures

### 1. Custom CNN Model

- Multiple convolutional blocks with batch normalization
- Dropout layers to prevent overfitting
- Dense layers for classification

### 2. Autoencoder + Classifier

- Unsupervised feature extraction using a convolutional autoencoder
- Features are then fed into a classifier network

### 3. Transfer Learning with DenseNet121

- Pre-trained DenseNet121 model as feature extractor
- Custom classification layers added on top

## Results

The notebook compares the performance of all three models using:

- Test accuracy
- Classification reports
- Confusion matrices
- Visual performance comparison

## Saving Models

The trained models are saved at:

```
/models/best_custom_model.keras
/models/best_custom_model_with_extracted_features.keras
/models/densenet121_best_model.keras
```

## Authors

- Rodolfo Borbon
