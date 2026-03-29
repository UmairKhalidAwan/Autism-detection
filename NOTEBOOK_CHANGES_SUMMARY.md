# Notebook Adaptation Summary

## Original Notebook
**File**: `Copy_of_ASD_Paper_1_Mamoona.ipynb`

The original notebook was designed for:
- Single CSV file with tabular data
- Features like Age, Sex, Ethnicity, Jaundice, Family history, etc.
- Binary classification (ASD traits: Yes/No)
- Google Colab file upload interface

## Your Dataset
Your dataset is **multimodal** with 4 different data types:
- **Images**: 100 facial images (`Child (1).jpg` to `Child (100).jpg`)
- **Voice**: 100 audio files (.wav format)
- **Motion**: 100 JSON files with body tracking data (head, hands, torso)
- **Physiological**: 100 CSV files with HR, GSR, TEMP measurements
- **Labels**: 4 classes (typical, mild_asd, moderate_asd, severe_asd)

## Key Changes Made

### 1. Data Loading
**Before**:
```python
uploaded = files.upload()  # Google Colab upload
df = pd.read_csv(next(iter(uploaded)))
```

**After**:
```python
df = pd.read_csv('autism_dataset_index.csv')  # Local file
```

### 2. Feature Extraction
**Before**: Used existing columns from CSV directly

**After**: Implemented 4 feature extraction functions:

#### a) Image Features (128 dimensions)
- Color histograms (RGB channels)
- Image intensity statistics (mean, std)
- Handles file path conversion (child_001.png → Child (1).jpg)

#### b) Voice Features (40 dimensions)
- MFCCs (Mel-Frequency Cepstral Coefficients): 13 coefficients × 2 (mean & std)
- Spectral centroid (frequency distribution center)
- Spectral rolloff (frequency below which 85% of energy exists)
- Zero-crossing rate (audio signal sign changes)

#### c) Motion Features (50 dimensions)
- Statistics per body part (head, left_hand, right_hand, torso):
  - Mean, std, min, max
  - 25th and 75th percentiles
- Stimming detection flag (from JSON)

#### d) Physiological Features (30 dimensions)
- Per signal (HR, GSR, TEMP):
  - Mean, std, min, max
  - 25th, 50th (median), 75th percentiles
  - Temporal change (last value - first value)

**Total: 248 features** combined from all modalities

### 3. Classification Task
**Before**: Binary classification (2 classes)
**After**: Multi-class classification (4 classes)

Neural network output layer changed:
```python
# Before
Dense(1, activation='sigmoid')  # Binary

# After
Dense(len(label_encoder.classes_), activation='softmax')  # Multi-class
```

Loss function changed:
```python
# Before
loss='binary_crossentropy'

# After
loss='sparse_categorical_crossentropy'
```

### 4. Label Handling
**Before**:
```python
df['ASD_traits'] = df['ASD_traits'].map({'Yes': 1, 'No': 0})
```

**After**:
```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
# Encodes: typical=0, mild_asd=1, moderate_asd=2, severe_asd=3
```

### 5. Additional Libraries
Added imports for multimedia processing:
```python
import librosa      # Audio processing
import cv2          # Image processing
import json         # Motion data
import os           # File operations
```

### 6. Error Handling
Added try-except blocks in all feature extraction functions to handle:
- Missing files
- Corrupted data
- Processing errors

Returns zero vectors if extraction fails to prevent crashes.

### 7. Visualization Enhancements

#### Cell 0 (Basic):
- Confusion matrices for each model
- Performance comparison bar chart
- Model accuracy summary

#### Cell 1 (Advanced):
- All Cell 0 visualizations PLUS:
- Neural network training history (accuracy & loss curves)
- SHAP feature importance plots
- LIME explanation for individual predictions
- Per-class fairness analysis

### 8. Feature Selection (Cell 1 only)
Added correlation-based feature selection:
```python
# Remove features with >0.95 correlation
corr_matrix = pd.DataFrame(X_train_resampled).corr().abs()
# Keeps only non-redundant features
```

### 9. Model Optimizations (Cell 1 only)

**Random Forest**:
- Increased trees: 100 → 200
- Deeper trees: max_depth 5 → 15
- Better generalization: min_samples_split, min_samples_leaf tuned

**XGBoost**:
- More regularization: reg_lambda, reg_alpha added
- Smaller learning rate: 0.1 → 0.01 (more stable)
- Increased trees: 100 → 150

**Neural Network**:
- Deeper: 3 layers → 4 layers
- Wider: 64 neurons → 256 neurons (first layer)
- Better regularization: Batch normalization added
- More dropout: 0.3 → 0.4
- Smaller learning rate: 0.001 → 0.0001

**Stacking Ensemble**:
- Combines predictions from RF, XGBoost, and SVM
- Meta-learner: Logistic Regression
- Often achieves best performance

### 10. Explainability Features (Cell 1 only)

#### SHAP (SHapley Additive exPlanations):
- Shows global feature importance
- Indicates which features contribute most to predictions
- Uses game theory to fairly attribute importance

#### LIME (Local Interpretable Model-agnostic Explanations):
- Explains individual predictions
- Shows why model predicted a specific class for a specific sample
- Useful for understanding model decisions on a case-by-case basis

#### Fairness Analysis:
- Checks if model performs equally across all severity classes
- Helps identify bias toward certain groups
- Important for clinical applications

## File Structure

### Before (Expected by Original Notebook):
```
- single_csv_file.csv  (uploaded via Colab)
```

### After (Required by Adapted Notebook):
```
Dr Noshina/
├── Copy_of_ASD_Paper_1_Mamoona.ipynb  (adapted notebook)
├── autism_dataset_index.csv
├── autism_dataset_metadata.json
├── requirements.txt  (NEW)
├── README_Dataset_Usage.md  (NEW)
├── NOTEBOOK_CHANGES_SUMMARY.md  (NEW - this file)
├── images/
│   └── Child (1-100).jpg
├── voice/
│   └── child_001-100.wav
├── motion/
│   └── child_001-100.json
└── physio/
    └── child_001-100.csv
```

## Models Comparison

### Cell 0: Basic Models
1. Random Forest
2. SVM
3. Gradient Boosting
4. XGBoost
5. Decision Tree
6. k-NN
7. Logistic Regression
8. Neural Network

### Cell 1: Advanced Models
1. Random Forest (Optimized)
2. XGBoost (Optimized)
3. Stacking Classifier (RF + XGBoost + SVM)
4. Neural Network (Deeper, Optimized)

Plus: SHAP, LIME, Fairness Analysis

## Expected Execution Time

### Feature Extraction (~5-10 minutes for 100 samples):
- Images: ~1-2 minutes
- Voice: ~5-8 minutes (slowest, librosa processing)
- Motion: ~30 seconds
- Physio: ~30 seconds

### Model Training:

#### Cell 0 (~5-10 minutes):
- RF, SVM, GBM, XGBoost, DT, k-NN, LogReg: ~2-3 minutes
- Neural Network (50 epochs): ~3-5 minutes

#### Cell 1 (~15-25 minutes):
- Feature selection: ~1 minute
- RF, XGBoost: ~3-4 minutes
- Stacking: ~5-7 minutes (trains 3 base models + meta-learner)
- Neural Network (100 epochs): ~7-10 minutes
- SHAP analysis: ~2-3 minutes
- LIME analysis: ~1 minute

**Total for Cell 1: ~20-30 minutes** (including feature extraction)

## Validation

The notebook uses:
1. **Train-test split**: 80% train, 20% test
2. **Stratified sampling**: Maintains class distribution
3. **SMOTE**: Balances training data
4. **Cross-validation**: Can be added if needed
5. **Multiple metrics**: Accuracy, confusion matrix, classification report

## Next Steps

1. **Run the adapted notebook** on your dataset
2. **Compare results** between Cell 0 (basic) and Cell 1 (advanced)
3. **Analyze SHAP values** to understand which modalities contribute most
4. **Use LIME** to understand specific predictions
5. **Check fairness analysis** for any bias
6. **Tune hyperparameters** further if needed
7. **Add cross-validation** for more robust evaluation
8. **Experiment with different fusion strategies** (early vs late fusion)

## Potential Improvements

1. **Deep Learning for Feature Extraction**:
   - Use pre-trained CNN for images (ResNet, VGG)
   - Use pre-trained models for audio (Wav2Vec)
   - Use LSTM/Transformer for motion sequences

2. **Attention Mechanisms**:
   - Learn which modality is most important per sample
   - Adaptive fusion based on data quality

3. **Ensemble Methods**:
   - Voting classifiers
   - Weighted averaging based on modality reliability

4. **Data Augmentation**:
   - Audio: pitch shifting, time stretching
   - Images: rotation, flip, color jitter
   - Motion: temporal warping

5. **Transfer Learning**:
   - Pre-train on larger autism datasets
   - Fine-tune on your specific data

## Conclusion

The notebook has been successfully adapted from a single-modality tabular dataset to a multi-modal dataset with images, voice, motion, and physiological data. All major machine learning models from the original paper are preserved and enhanced with:

- ✅ Multimodal feature extraction
- ✅ Multi-class classification (4 severity levels)
- ✅ Model optimization
- ✅ Ensemble methods (stacking)
- ✅ Explainability (SHAP & LIME)
- ✅ Fairness analysis
- ✅ Comprehensive visualization

The adapted notebook is now ready to run on your autism dataset!

