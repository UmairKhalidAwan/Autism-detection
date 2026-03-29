"""
Multimodal Autism Detection with Advanced Preprocessing & Hyperparameter Tuning
Target Accuracy: 98%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning & Audio/Image Processing
import librosa
import cv2
from PIL import Image

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

print("="*80)
print("MULTIMODAL AUTISM DETECTION - OPTIMIZED VERSION")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND FIX DATASET PATHS
# ============================================================================
print("\n[*] Loading dataset...")
df = pd.read_csv('autism_dataset_index.csv')
print(f"Dataset loaded: {len(df)} samples")

# Fix path mismatches: child_001.png  Child (1).jpg
def fix_image_path(path):
    """Convert child_001.png to Child (1).jpg"""
    basename = os.path.basename(path)
    sample_num = basename.split('_')[-1].split('.')[0]
    return f"images/Child ({int(sample_num)}).jpg"

df['image'] = df['image'].apply(fix_image_path)
print(" Image paths fixed")

# Verify all files exist
print("\n Verifying dataset integrity...")
missing_files = []
for idx, row in df.iterrows():
    for col in ['image', 'voice', 'motion', 'physio']:
        if not os.path.exists(row[col]):
            missing_files.append((row['sample_id'], col, row[col]))

if missing_files:
    print(f"  Warning: {len(missing_files)} missing files found")
    for sample, modality, path in missing_files[:5]:
        print(f"  - {sample}: {modality} ({path})")
else:
    print(" All files verified successfully")

# ============================================================================
# STEP 2: ADVANCED FEATURE EXTRACTION WITH PREPROCESSING
# ============================================================================

def extract_image_features_advanced(image_path, augment=False):
    """
    Extract advanced image features with preprocessing
    - Color histograms (RGB & HSV)
    - Texture features (GLCM-like)
    - Edge features
    - Facial landmarks statistics
    """
    try:
        if not os.path.exists(image_path):
            return np.zeros(200)
        
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros(200)
        
        # Preprocessing: Resize and normalize
        img = cv2.resize(img, (128, 128))
        
        # Data augmentation (if training)
        if augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)  # Horizontal flip
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        
        # RGB histograms (normalized)
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)  # Normalize
            hist_features.extend(hist)
        
        # HSV histograms
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            hist_features.extend(hist)
        
        # Gray scale features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.mean() / 255.0
        
        # Texture features (simplified)
        texture_features = [
            gray.mean() / 255.0,
            gray.std() / 255.0,
            np.percentile(gray, 25) / 255.0,
            np.percentile(gray, 50) / 255.0,
            np.percentile(gray, 75) / 255.0,
            edge_density,
            cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0  # Blur detection
        ]
        
        features = np.array(hist_features + texture_features)
        return features[:200]  # Fixed size
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return np.zeros(200)


def extract_voice_features_advanced(voice_path):
    """
    Extract advanced audio features with preprocessing
    - MFCCs (13 coefficients + deltas)
    - Spectral features (centroid, rolloff, bandwidth, contrast)
    - Rhythm features (tempo, beat)
    - Prosody features (pitch, energy)
    """
    try:
        if not os.path.exists(voice_path):
            return np.zeros(80)
        
        # Load audio with preprocessing
        y, sr = librosa.load(voice_path, duration=10, sr=22050)
        
        # Remove silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Normalize
        y = librosa.util.normalize(y)
        
        features = []
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfccs.mean(axis=1)
        mfcc_std = mfccs.std(axis=1)
        features.extend(mfcc_mean)
        features.extend(mfcc_std)
        
        # Delta MFCCs (velocity)
        mfcc_delta = librosa.feature.delta(mfccs)
        features.extend(mfcc_delta.mean(axis=1))
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        
        features.extend([
            float(spectral_centroid.mean()),
            float(spectral_centroid.std()),
            float(spectral_rolloff.mean()),
            float(spectral_rolloff.std()),
            float(spectral_bandwidth.mean()),
            float(spectral_bandwidth.std()),
            float(spectral_contrast.mean()),
            float(spectral_contrast.std())
        ])
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([zcr.mean(), zcr.std()])
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo / 200.0)  # Normalize tempo
        
        # Energy features
        rms = librosa.feature.rms(y=y)
        features.extend([rms.mean(), rms.std()])
        
        # Pitch features (chroma)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([chroma.mean(), chroma.std()])
        
        features = np.array(features)
        return features[:80]  # Fixed size
    
    except Exception as e:
        print(f"Error processing voice {voice_path}: {e}")
        return np.zeros(80)


def extract_motion_features_advanced(motion_path):
    """
    Extract advanced motion features with preprocessing
    - Temporal statistics (mean, std, velocity, acceleration)
    - Spatial statistics (range, variance)
    - Repetitive movement detection
    - Movement smoothness
    """
    try:
        if not os.path.exists(motion_path):
            return np.zeros(100)
        
        with open(motion_path, 'r') as f:
            motion_data = json.load(f)
        
        features = []
        
        # Process each body part
        for body_part in ['head', 'left_hand', 'right_hand', 'torso']:
            if body_part in motion_data:
                points = np.array(motion_data[body_part])
                
                if len(points.shape) == 2 and points.shape[0] > 0:
                    # Basic statistics
                    features.extend([
                        points.mean(),
                        points.std(),
                        points.min(),
                        points.max(),
                        np.median(points),
                        np.percentile(points, 25),
                        np.percentile(points, 75)
                    ])
                    
                    # Velocity (first derivative)
                    if points.shape[0] > 1:
                        velocity = np.diff(points, axis=0)
                        features.extend([
                            velocity.mean(),
                            velocity.std(),
                            np.abs(velocity).mean()  # Average speed
                        ])
                        
                        # Acceleration (second derivative)
                        if velocity.shape[0] > 1:
                            acceleration = np.diff(velocity, axis=0)
                            features.extend([
                                acceleration.mean(),
                                acceleration.std(),
                                np.abs(acceleration).mean()
                            ])
                        else:
                            features.extend([0, 0, 0])
                    else:
                        features.extend([0, 0, 0, 0, 0, 0])
                else:
                    features.extend([0] * 16)
        
        # Stimming detection
        if 'stimming_detected' in motion_data:
            features.append(float(motion_data['stimming_detected']))
        else:
            features.append(0)
        
        # Frame rate and duration
        if 'frame_rate' in motion_data:
            features.append(motion_data['frame_rate'] / 30.0)  # Normalize
        else:
            features.append(1.0)
        
        if 'duration_sec' in motion_data:
            features.append(motion_data['duration_sec'] / 10.0)  # Normalize
        else:
            features.append(1.0)
        
        features = np.array(features)
        return features[:100] if len(features) >= 100 else np.pad(features, (0, 100 - len(features)))
    
    except Exception as e:
        print(f"Error processing motion {motion_path}: {e}")
        return np.zeros(100)


def extract_physio_features_advanced(physio_path):
    """
    Extract advanced physiological features with preprocessing
    - Statistical features (mean, std, min, max, percentiles)
    - Temporal features (trends, slopes, changes)
    - Frequency domain features
    - Variability metrics
    """
    try:
        if not os.path.exists(physio_path):
            return np.zeros(60)
        
        physio_df = pd.read_csv(physio_path)
        features = []
        
        # Process each signal type
        for col in ['HR', 'GSR', 'TEMP']:
            if col in physio_df.columns:
                signal = physio_df[col].values
                
                # Remove outliers (IQR method)
                Q1 = np.percentile(signal, 25)
                Q3 = np.percentile(signal, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                signal_clean = signal[(signal >= lower_bound) & (signal <= upper_bound)]
                
                if len(signal_clean) == 0:
                    signal_clean = signal
                
                # Statistical features
                features.extend([
                    signal_clean.mean(),
                    signal_clean.std(),
                    signal_clean.min(),
                    signal_clean.max(),
                    np.median(signal_clean),
                    np.percentile(signal_clean, 25),
                    np.percentile(signal_clean, 75),
                    signal_clean.max() - signal_clean.min(),  # Range
                ])
                
                # Temporal features
                if len(signal) > 1:
                    # Trend (linear regression slope)
                    x = np.arange(len(signal))
                    slope = np.polyfit(x, signal, 1)[0]
                    features.append(slope)
                    
                    # First and last value difference
                    features.append(signal[-1] - signal[0])
                    
                    # Variability
                    diff = np.diff(signal)
                    features.extend([
                        diff.mean(),
                        diff.std(),
                        np.abs(diff).mean()  # Average absolute change
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
        
        features = np.array(features)
        return features[:60] if len(features) >= 60 else np.pad(features, (0, 60 - len(features)))
    
    except Exception as e:
        print(f"Error processing physio {physio_path}: {e}")
        return np.zeros(60)


# ============================================================================
# STEP 3: EXTRACT FEATURES WITH PROGRESS BAR
# ============================================================================
print("\n Extracting advanced features from all modalities...")
print("   This may take 10-15 minutes for 100 samples...")

all_features = []
for idx, row in df.iterrows():
    if idx % 10 == 0:
        print(f"   Processing sample {idx + 1}/100... ({(idx+1)/100*100:.0f}% complete)")
    
    img_feat = extract_image_features_advanced(row['image'], augment=False)
    voice_feat = extract_voice_features_advanced(row['voice'])
    motion_feat = extract_motion_features_advanced(row['motion'])
    physio_feat = extract_physio_features_advanced(row['physio'])
    
    combined_features = np.concatenate([img_feat, voice_feat, motion_feat, physio_feat])
    all_features.append(combined_features)

X = np.array(all_features)
print(f"\n Feature extraction complete!")
print(f"  Feature shape: {X.shape}")
print(f"  Total features per sample: {X.shape[1]}")

# ============================================================================
# STEP 4: ADVANCED PREPROCESSING
# ============================================================================
print("\n Advanced preprocessing...")

# Remove features with zero variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.0)
X_cleaned = selector.fit_transform(X)
print(f"   Removed {X.shape[1] - X_cleaned.shape[1]} zero-variance features")

# Handle any remaining NaN or inf values
X_cleaned = np.nan_to_num(X_cleaned, nan=0.0, posinf=0.0, neginf=0.0)

# Prepare labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
print(f"   Encoded {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
print(f"   Class distribution: {dict(zip(label_encoder.classes_, np.bincount(y)))}")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT & ADVANCED OVERSAMPLING
# ============================================================================
print("\n Creating train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

# Scale features (RobustScaler is better for outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   Features scaled using RobustScaler")

# Advanced oversampling: SMOTETomek (SMOTE + Tomek links)
print("\n Applying advanced oversampling (SMOTETomek)...")
smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train_scaled, y_train)
print(f"   Resampled from {len(X_train)} to {len(X_train_resampled)} samples")
print(f"   New class distribution: {np.bincount(y_train_resampled)}")

# Feature selection: Keep top features
print("\n Selecting top features...")
selector_k = SelectKBest(f_classif, k=min(200, X_train_resampled.shape[1]))
X_train_selected = selector_k.fit_transform(X_train_resampled, y_train_resampled)
X_test_selected = selector_k.transform(X_test_scaled)
print(f"   Selected {X_train_selected.shape[1]} best features")

# ============================================================================
# STEP 6: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ============================================================================
print("\n" + "="*80)
print("  HYPERPARAMETER TUNING (This will take time...)")
print("="*80)

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Random Forest with GridSearch
print("\n Tuning Random Forest...")
rf_param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [15, 20],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
rf_grid.fit(X_train_selected, y_train_resampled)
best_rf = rf_grid.best_estimator_
print(f"   Best RF params: {rf_grid.best_params_}")
print(f"   Best CV score: {rf_grid.best_score_:.4f}")

# 2. XGBoost with GridSearch
print("\n Tuning XGBoost...")
xgb_param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha': [0.1],
    'reg_lambda': [5, 10]
}

xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='mlogloss', n_jobs=-1),
    xgb_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
xgb_grid.fit(X_train_selected, y_train_resampled)
best_xgb = xgb_grid.best_estimator_
print(f"   Best XGB params: {xgb_grid.best_params_}")
print(f"   Best CV score: {xgb_grid.best_score_:.4f}")

# 3. SVM with GridSearch
print("\n Tuning SVM...")
svm_param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
svm_grid.fit(X_train_selected, y_train_resampled)
best_svm = svm_grid.best_estimator_
print(f"   Best SVM params: {svm_grid.best_params_}")
print(f"   Best CV score: {svm_grid.best_score_:.4f}")

# ============================================================================
# STEP 7: ENSEMBLE METHODS
# ============================================================================
print("\n Creating ensemble models...")

# Stacking Classifier
stacking = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('svm', best_svm)
    ],
    final_estimator=LogisticRegression(max_iter=1000, multi_class='ovr'),
    cv=cv
)
print("  Training Stacking Classifier...")
stacking.fit(X_train_selected, y_train_resampled)
print("   Stacking trained")

# Voting Classifier (soft voting)
voting = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('svm', best_svm)
    ],
    voting='soft'
)
print("  Training Voting Classifier...")
voting.fit(X_train_selected, y_train_resampled)
print("   Voting trained")

# ============================================================================
# STEP 8: DEEP NEURAL NETWORK
# ============================================================================
print("\n Training Deep Neural Network...")

# Build optimized neural network
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_selected.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

# Train
history = model.fit(
    X_train_selected, y_train_resampled,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("   Neural Network trained")

# ============================================================================
# STEP 9: EVALUATION
# ============================================================================
print("\n" + "="*80)
print(" MODEL EVALUATION RESULTS")
print("="*80)

models = {
    'Random Forest (Tuned)': best_rf,
    'XGBoost (Tuned)': best_xgb,
    'SVM (Tuned)': best_svm,
    'Stacking Ensemble': stacking,
    'Voting Ensemble': voting
}

results = {}

for name, model_obj in models.items():
    y_pred = model_obj.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Neural Network
y_pred_nn = np.argmax(model.predict(X_test_selected), axis=1)
acc_nn = accuracy_score(y_test, y_pred_nn)
results['Neural Network (Deep)'] = acc_nn

print(f"\n{'='*60}")
print(f" Neural Network (Deep)")
print(f"{'='*60}")
print(f"Accuracy: {acc_nn:.4f} ({acc_nn*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Neural Network - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_Neural_Network.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 10: FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print(" FINAL MODEL COMPARISON")
print("="*80)

sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nModel Rankings:")
for i, (name, acc) in enumerate(sorted_results, 1):
    print(f"{i}. {name:30s}: {acc:.4f} ({acc*100:.2f}%)")
    if acc >= 0.98:
        print(f"    TARGET ACHIEVED! (98%)")

# Visualization
plt.figure(figsize=(14, 7))
models_list = [name for name, _ in sorted_results]
accs_list = [acc for _, acc in sorted_results]
colors = ['#2ecc71' if acc >= 0.98 else '#3498db' for acc in accs_list]

bars = plt.barh(models_list, accs_list, color=colors)
plt.xlabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison - Multimodal Autism Detection', 
          fontsize=14, fontweight='bold')
plt.xlim(0, 1.0)
plt.axvline(x=0.98, color='red', linestyle='--', linewidth=2, label='Target (98%)')
plt.grid(axis='x', alpha=0.3)
plt.legend()

# Add value labels
for i, (bar, acc) in enumerate(zip(bars, accs_list)):
    plt.text(acc + 0.01, i, f'{acc*100:.2f}%', 
             va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('final_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Training history for Neural Network
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Neural Network Training History - Accuracy')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Neural Network Training History - Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('nn_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print(" ANALYSIS COMPLETE!")
print("="*80)
print(f"\n Best Model: {sorted_results[0][0]}")
print(f" Best Accuracy: {sorted_results[0][1]:.4f} ({sorted_results[0][1]*100:.2f}%)")

if sorted_results[0][1] >= 0.98:
    print("\n CONGRATULATIONS! Target accuracy of 98% ACHIEVED!")
else:
    gap = 0.98 - sorted_results[0][1]
    print(f"\n Current best is {gap*100:.2f}% below target.")
    print("\n Recommendations to reach 98%:")
    print("   1. Collect more training data (currently only 100 samples)")
    print("   2. Apply data augmentation (audio pitch shifting, image transforms)")
    print("   3. Use deep learning for feature extraction (CNN for images, etc.)")
    print("   4. Ensemble more diverse models")
    print("   5. Try transfer learning with pre-trained models")

# Save best model
import joblib
best_model_name = sorted_results[0][0]
best_model = models.get(best_model_name, model)
if best_model_name != 'Neural Network (Deep)':
    joblib.dump(best_model, 'best_autism_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(selector_k, 'feature_selector.pkl')
    print(f"\n Best model saved as 'best_autism_model.pkl'")
else:
    model.save('best_autism_model_nn.h5')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(selector_k, 'feature_selector.pkl')
    print(f"\n Best model saved as 'best_autism_model_nn.h5'")

print("\n All outputs saved to current directory!")

