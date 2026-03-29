# Multimodal Autism Detection - Dataset Usage Guide

## Overview
This notebook has been adapted to work with your multimodal autism dataset that includes:
- **Images**: Facial images of children (100 samples)
- **Voice**: Audio recordings (.wav files)
- **Motion**: Body movement tracking data (JSON files with head, hands, torso coordinates)
- **Physiological**: HR, GSR, and Temperature measurements (CSV files)

## Dataset Structure
```
Dr Noshina/
├── autism_dataset_index.csv       # Main index file
├── autism_dataset_metadata.json   # Dataset metadata
├── images/                         # Facial images
│   ├── Child (1).jpg
│   ├── Child (2).jpg
│   └── ...
├── voice/                          # Voice recordings
│   ├── child_001.wav
│   ├── child_002.wav
│   └── ...
├── motion/                         # Motion tracking data
│   ├── child_001.json
│   ├── child_002.json
│   └── ...
└── physio/                         # Physiological signals
    ├── child_001.csv
    ├── child_002.csv
    └── ...
```

## Labels
The dataset has 4 severity classes:
- `typical`: Neurotypical children (no ASD)
- `mild_asd`: Mild autism spectrum disorder
- `moderate_asd`: Moderate autism spectrum disorder
- `severe_asd`: Severe autism spectrum disorder

## Notebook Cells

### Cell 0: Basic Multimodal Analysis
This cell includes:
1. **Feature Extraction** from all modalities:
   - Image features: Color histograms, intensity statistics
   - Voice features: MFCCs, spectral features, zero-crossing rate
   - Motion features: Body part movement statistics, stimming detection
   - Physio features: HR/GSR/TEMP statistics and temporal changes

2. **Machine Learning Models**:
   - Random Forest
   - Support Vector Machine (SVM)
   - Gradient Boosting
   - XGBoost
   - Decision Tree
   - k-Nearest Neighbors
   - Logistic Regression
   - Neural Network

3. **Outputs**:
   - Confusion matrices for each model
   - Performance comparison bar chart
   - Classification reports

### Cell 1: Advanced Analysis with Explainability
This cell includes everything from Cell 0 plus:
1. **Feature Selection**: Removes highly correlated features (>0.95 correlation)
2. **Optimized Hyperparameters**: Better-tuned models
3. **Stacking Ensemble**: Combines RF, XGBoost, and SVM
4. **Enhanced Neural Network**: Deeper architecture with batch normalization
5. **Explainability Tools**:
   - **SHAP**: Shows which features are most important globally
   - **LIME**: Explains individual predictions
6. **Fairness Analysis**: Checks model performance across different severity classes

## Running the Notebook

### Prerequisites
Install required packages:
```bash
pip install xgboost shap lime imbalanced-learn tensorflow scikit-learn pandas numpy matplotlib seaborn librosa opencv-python pillow
```

### Steps
1. Make sure all data files are in the correct folders
2. Run Cell 0 for basic analysis (faster, ~5-10 minutes)
   OR
3. Run Cell 1 for advanced analysis with explainability (slower, ~15-20 minutes)

## Key Features

### Multimodal Feature Fusion
The notebook combines features from all 4 modalities:
- **128** image features
- **40** voice features
- **50** motion features
- **30** physiological features
- **Total: 248** features per sample

### SMOTE for Class Imbalance
Since the dataset has imbalanced classes, SMOTE (Synthetic Minority Over-sampling Technique) is applied to balance the training data.

### Visualization
The notebook generates:
- Confusion matrices for each model
- Performance comparison charts
- Neural network training history
- SHAP feature importance plots
- LIME explanations for individual predictions

## Interpreting Results

### Accuracy
- Shows overall classification performance
- Values closer to 1.0 are better

### Confusion Matrix
- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

### SHAP Values
- Positive values: Feature pushes prediction toward that class
- Negative values: Feature pushes prediction away from that class
- Larger magnitude: More important feature

### Per-Class Performance
The fairness analysis shows if the model performs consistently across severity levels or has bias toward certain classes.

## Troubleshooting

### Missing Files
If you see "Warning: Image not found" or similar:
- Check that file naming matches the index CSV
- The code expects images as "Child (1).jpg" format
- Voice/motion/physio files should match the paths in the CSV

### Memory Issues
If you run out of memory:
- Reduce the neural network epochs (from 100 to 50)
- Reduce the number of SMOTE samples
- Use a smaller subset of the dataset for testing

### Slow Execution
Audio feature extraction (librosa) can be slow:
- It processes 10 seconds per audio file
- For 100 samples, expect ~5-10 minutes for audio alone
- Consider reducing audio duration in `extract_voice_features()`

## Customization

### Adjust Neural Network Architecture
In both cells, you can modify:
- Number of layers
- Layer sizes (neurons per layer)
- Dropout rates
- Learning rate
- Number of epochs

### Change Model Hyperparameters
Each model has tunable parameters:
- `n_estimators`: Number of trees (RF, XGBoost, GBM)
- `max_depth`: Tree depth
- `learning_rate`: Step size for gradient descent
- `C`, `gamma`: SVM parameters

### Select Different Models
Comment out models you don't want to train to save time.

## Citation
If you use this adapted notebook in your research, please cite:
- The original paper the notebook was based on
- The multimodal autism dataset
- The machine learning libraries used (scikit-learn, XGBoost, TensorFlow, SHAP, LIME)

## Contact
For questions about the adapted notebook, refer to the original paper or the dataset documentation.

