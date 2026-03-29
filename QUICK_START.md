# Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Required Packages
Open a terminal in this directory and run:
```bash
pip install -r requirements.txt
```

This will install all necessary libraries (TensorFlow, scikit-learn, librosa, etc.)

### Step 2: Verify Dataset Structure
Make sure your folder structure looks like this:
```
Dr Noshina/
├── Copy_of_ASD_Paper_1_Mamoona.ipynb  ← Open this
├── autism_dataset_index.csv           ← Must be here
├── images/
│   └── Child (1).jpg to Child (100).jpg
├── voice/
│   └── child_001.wav to child_100.wav
├── motion/
│   └── child_001.json to child_100.json
└── physio/
    └── child_001.csv to child_100.csv
```

### Step 3: Run the Notebook
Open `Copy_of_ASD_Paper_1_Mamoona.ipynb` in Jupyter:
```bash
jupyter notebook Copy_of_ASD_Paper_1_Mamoona.ipynb
```

Or if using JupyterLab:
```bash
jupyter lab Copy_of_ASD_Paper_1_Mamoona.ipynb
```

Or use Google Colab:
1. Upload the notebook to Google Drive
2. Right-click → Open with → Google Colaboratory
3. Upload your dataset folders to Colab environment

## Which Cell to Run?

### Option 1: Cell 0 - Basic Analysis (Faster)
**Run this first if**:
- You want quick results (~10-15 minutes)
- You want to see all 8 models
- You just want accuracy comparisons

**Includes**:
- ✅ All 8 ML models (RF, SVM, XGBoost, etc.)
- ✅ Confusion matrices
- ✅ Performance comparison chart
- ✅ Classification reports

### Option 2: Cell 1 - Advanced Analysis (Better Results)
**Run this if**:
- You want the best accuracy (~20-30 minutes)
- You want to understand feature importance
- You need explainable AI (SHAP/LIME)
- You want fairness analysis

**Includes**:
- ✅ Optimized models (better hyperparameters)
- ✅ Stacking ensemble (combines multiple models)
- ✅ Feature selection (removes redundant features)
- ✅ SHAP feature importance plots
- ✅ LIME individual prediction explanations
- ✅ Per-class fairness analysis
- ✅ Neural network training curves

## Expected Results

### Typical Output:
```
Loading dataset index...
Dataset loaded: 100 samples

Extracting features from all modalities...
Processing sample 1/100...
Processing sample 11/100...
...
Feature extraction complete. Shape: (100, 248)

Label distribution: {'mild_asd': 25, 'moderate_asd': 25, 'severe_asd': 25, 'typical': 25}

Train size: 80, Test size: 20
Applying SMOTE...
Resampled train size: 200

🔹 Training Random Forest:
Accuracy: 0.8500
              precision    recall  f1-score   support
    mild_asd       0.83      0.83      0.83         6
moderate_asd       0.86      0.86      0.86         7
  severe_asd       0.85      0.86      0.85         7
     typical       1.00      1.00      1.00         6

[Confusion matrix displayed]
```

### Performance Benchmarks:
Based on the multimodal nature of your dataset, you can expect:
- **Good models**: 70-80% accuracy
- **Very good models**: 80-90% accuracy
- **Excellent models**: 90%+ accuracy

The stacking ensemble in Cell 1 typically performs best.

## Troubleshooting

### Error: "Module not found"
```bash
pip install <module_name>
# or
pip install -r requirements.txt
```

### Error: "Image not found"
Check that:
- Images are named exactly: `Child (1).jpg`, `Child (2).jpg`, etc.
- They're in the `images/` folder
- The path in `autism_dataset_index.csv` is correct

### Error: "Audio file not found"
Check that:
- Audio files are named: `child_001.wav`, `child_002.wav`, etc.
- They're in the `voice/` folder
- File extensions are `.wav` (not `.mp3` or other formats)

### Error: "Out of memory"
Reduce the dataset size for testing:
```python
# Add this after loading the dataset
df = df.sample(n=50, random_state=42)  # Use only 50 samples
```

Or reduce neural network complexity:
```python
# Reduce epochs
epochs=25  # instead of 50 or 100

# Reduce batch size
batch_size=16  # instead of 32 or 64
```

### Slow Execution
Voice feature extraction is the slowest part (librosa).
To speed up, reduce audio duration:
```python
def extract_voice_features(voice_path):
    y, sr = librosa.load(voice_path, duration=5)  # Use 5 seconds instead of 10
    # ... rest of the code
```

## Understanding the Output

### Confusion Matrix
```
               Predicted
           mild  mod  sev  typ
Actual mild  [5]   1    0    0
       mod    1  [6]   0    0
       sev    0   1  [6]   0
       typ    0   0    0  [6]
```
- **Diagonal** (in brackets): Correct predictions
- **Off-diagonal**: Misclassifications

### Classification Report
```
              precision    recall  f1-score
    mild_asd       0.83      0.83      0.83
moderate_asd       0.86      0.86      0.86
  severe_asd       0.85      0.86      0.85
     typical       1.00      1.00      1.00
```
- **Precision**: Of all predicted as this class, how many were correct?
- **Recall**: Of all actual samples of this class, how many did we find?
- **F1-score**: Harmonic mean of precision and recall

### SHAP Plot (Cell 1 only)
Shows which features matter most:
- **Top features**: Most important for classification
- **Color**: Feature value (red=high, blue=low)
- **X-axis**: SHAP value (impact on prediction)

### LIME Plot (Cell 1 only)
Explains a single prediction:
- **Green bars**: Features that support the prediction
- **Red bars**: Features that oppose the prediction
- **Length**: Strength of contribution

## Next Steps After Running

1. **Compare models**: Which performs best on your data?
2. **Check confusion**: Which classes are confused with each other?
3. **Analyze SHAP**: Which modality (image/voice/motion/physio) matters most?
4. **Inspect errors**: Look at misclassified samples
5. **Tune hyperparameters**: Can you improve accuracy?
6. **Add more data**: Collect more samples if possible
7. **Try deep learning**: Use CNNs for images, RNNs for sequences

## Getting Help

If something doesn't work:
1. Check the error message carefully
2. Read `README_Dataset_Usage.md` for detailed info
3. Read `NOTEBOOK_CHANGES_SUMMARY.md` to understand what changed
4. Check that your file structure matches exactly
5. Verify all data files exist and are readable

## Performance Tips

### For Faster Execution:
1. Use fewer samples for testing (e.g., 50 instead of 100)
2. Reduce neural network epochs (25 instead of 50/100)
3. Skip slower models (SVM, k-NN)
4. Use only Cell 0 (skip Cell 1)

### For Better Accuracy:
1. Use Cell 1 (advanced, optimized models)
2. Increase neural network epochs (100 or more)
3. Add more data samples
4. Try the stacking ensemble
5. Tune hyperparameters using GridSearchCV

## Common Questions

**Q: Which cell should I run?**
A: Run Cell 0 first for quick results. If you need better accuracy and explainability, run Cell 1.

**Q: Can I run both cells?**
A: Yes, but run them separately (not at the same time). They're independent.

**Q: How long will it take?**
A: Cell 0: ~10-15 minutes. Cell 1: ~20-30 minutes (including feature extraction).

**Q: What accuracy should I expect?**
A: 70-90% is typical for multimodal classification with 100 samples.

**Q: Can I use only some modalities?**
A: Yes, comment out the feature extraction you don't want and adjust the concatenation:
```python
# Use only image and voice features
combined_features = np.concatenate([img_feat, voice_feat])
```

**Q: How do I save the trained model?**
A: Add this after training:
```python
import joblib
joblib.dump(xgb, 'autism_xgboost_model.pkl')

# Load later
model = joblib.load('autism_xgboost_model.pkl')
```

## Success Indicators

You'll know it's working when you see:
- ✅ "Feature extraction complete" with shape (100, 248)
- ✅ All models train without errors
- ✅ Confusion matrices displayed
- ✅ Performance comparison chart
- ✅ Accuracy scores between 0.60-1.00

## Happy Coding! 🚀

Your notebook is now ready to analyze multimodal autism data. Good luck with your research!

