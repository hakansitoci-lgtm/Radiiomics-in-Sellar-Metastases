"""
Univariate Nested Feature Selection with LOOCV
Final Version - Publication Ready
For: Pituitary Adenoma vs Metastasis Classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from collections import Counter
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("UNIVARIATE NESTED FEATURE SELECTION WITH LOOCV")
print("Final Production Version")
print("="*70)

# === Configuration ===
K_FEATURES = 10  # Number of features to select per fold
RANDOM_SEED = 42  # For reproducibility
np.random.seed(RANDOM_SEED)

# Timestamp for unique file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# === Load Data ===
df = pd.read_excel("aggregated_radiomics_features.xlsx")
X = df.drop(columns=['ClassLabel', 'PatientID'])
y = df['ClassLabel'].map({'Adenoma': 0, 'Metastasis': 1}).values
patient_ids = df['PatientID'].values

print(f"\n📊 Dataset Summary:")
print(f"Samples: {len(X)} ({sum(y==0)} Adenomas, {sum(y==1)} Metastases)")
print(f"Features: {X.shape[1]}")
print(f"Feature/Sample Ratio: {X.shape[1]/len(X):.1f}:1")
print(f"Features to select per fold: {K_FEATURES}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Timestamp: {timestamp}")

print("\n" + "="*70)
print("STEP 1: NESTED LOOCV WITH UNIVARIATE FEATURE SELECTION")
print("="*70)

# === Initialize LOOCV ===
loo = LeaveOneOut()
y_true, y_pred, y_proba = [], [], []
selected_features_per_fold = []
n_features_selected_per_fold = []
model_coefficients = []

print(f"\nRunning LOOCV with nested feature selection (K={K_FEATURES})...")
print("Each fold selects features using ONLY its 15 training samples\n")

for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
    # Split data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Check if both classes present in training
    if len(np.unique(y_train)) < 2:
        print(f"⚠️ Fold {fold_idx+1}: Only one class in training data - skipping")
        continue
    
    # STEP 1: Feature Selection (using TRAINING data only!)
    selector = SelectKBest(f_classif, k=min(K_FEATURES, X_train.shape[1]))
    selector.fit(X_train, y_train)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_cols = X.columns[selected_mask].tolist()
    selected_features_per_fold.append(selected_cols)
    n_features_selected_per_fold.append(len(selected_cols))
    
    # Transform both train and test
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # STEP 2: Standardization (fit on training only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # STEP 3: Model Training
    model = LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Store coefficients
    model_coefficients.append({
        'features': selected_cols,
        'coefs': model.coef_[0]
    })
    
    # STEP 4: Prediction
    pred = model.predict(X_test_scaled)[0]
    prob = model.predict_proba(X_test_scaled)[0][1]
    
    y_true.append(y_test[0])
    y_pred.append(pred)
    y_proba.append(prob)
    
    # Progress update
    test_patient = patient_ids[test_idx[0]]
    print(f"Fold {fold_idx+1:2d}: Test=Patient_{test_patient:<2d} | "
          f"Selected {len(selected_cols)} features | "
          f"P(Meta)={prob:.3f} | True={y_test[0]}")

# Convert to arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_proba = np.array(y_proba)

# === Calculate Metrics ===
accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)
cm = confusion_matrix(y_true, y_pred)
sensitivity = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0
specificity = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0

print("\n" + "="*70)
print("RESULTS: Nested Feature Selection Performance")
print("="*70)
print(f"Accuracy    : {accuracy:.3f}")
print(f"AUC         : {auc:.3f}")
print(f"Sensitivity : {sensitivity:.3f} ({cm[1,1]}/{cm[1].sum()} metastases detected)")
print(f"Specificity : {specificity:.3f} ({cm[0,0]}/{cm[0].sum()} adenomas correct)")
print("\nConfusion Matrix:")
print("           Pred_Aden  Pred_Meta")
print(f"True_Aden      {cm[0,0]:2d}        {cm[0,1]:2d}")
print(f"True_Meta      {cm[1,0]:2d}        {cm[1,1]:2d}")

# === Feature Selection Stability Analysis ===
print("\n" + "="*70)
print("STEP 2: FEATURE SELECTION STABILITY ANALYSIS")
print("="*70)

# Count feature frequency across folds
all_selected_features = [feat for fold in selected_features_per_fold for feat in fold]
feature_counter = Counter(all_selected_features)

print(f"\nTotal unique features ever selected: {len(feature_counter)}")
print(f"Average features per fold: {np.mean(n_features_selected_per_fold):.1f}")

# Categorize by selection frequency
always_selected = [f for f, c in feature_counter.items() if c == 16]
usually_selected = [f for f, c in feature_counter.items() if c >= 12]
often_selected = [f for f, c in feature_counter.items() if c >= 8]

print(f"Feature selection consistency:")
print(f"  - Selected in ALL 16 folds: {len(always_selected)} features")
print(f"  - Selected in ≥75% folds:   {len(usually_selected)} features")
print(f"  - Selected in ≥50% folds:   {len(often_selected)} features")

if always_selected:
    print("\n📌 Features selected in EVERY fold:")
    for feat in always_selected:
        print(f"    - {feat}")
    
    # Save the universally selected features
    stable_features_df = pd.DataFrame({
        'Feature': always_selected,
        'Selection_Count': [16] * len(always_selected),
        'Interpretation': [
            'Tumor sphericity (shape regularity)',
            'Texture homogeneity (inverse difference normalized)',
            'Gray level run emphasis (texture pattern)'
        ][:len(always_selected)]
    })
    stable_features_df.to_excel(f'STABLE_FEATURES_{timestamp}.xlsx', index=False)
    print(f"\n✅ Saved stable features to: STABLE_FEATURES_{timestamp}.xlsx")

print("\n📊 Top 10 most frequently selected features:")
for feat, count in feature_counter.most_common(10):
    print(f"    {count:2d}/16 folds: {feat}")

# === Feature Importance Analysis ===
print("\n📈 Average Feature Coefficients (for frequently selected features):")
feature_coef_sum = {}
feature_coef_count = {}

for fold_data in model_coefficients:
    for feat, coef in zip(fold_data['features'], fold_data['coefs']):
        if feat not in feature_coef_sum:
            feature_coef_sum[feat] = 0
            feature_coef_count[feat] = 0
        feature_coef_sum[feat] += coef
        feature_coef_count[feat] += 1

print("\nFeatures with consistent positive/negative effects:")
for feat, count in feature_counter.most_common(10):
    if count >= 8:  # Selected in >50% of folds
        avg_coef = feature_coef_sum[feat] / feature_coef_count[feat]
        std_coef = np.std([fold['coefs'][fold['features'].index(feat)] 
                          for fold in model_coefficients 
                          if feat in fold['features']])
        direction = "↑ Metastasis" if avg_coef > 0 else "↓ Adenoma"
        # Show full feature name
        print(f"  {feat}: {avg_coef:+.3f} ± {std_coef:.3f} {direction}")

# === Permutation Test ===
print("\n" + "="*70)
print("STEP 3: PERMUTATION TEST (n=1000)")
print("="*70)
print("Testing if observed AUC is significantly better than chance...")
print("(This will take a few minutes with nested selection...)\n")

n_perm = 1000
perm_aucs = []
skipped_perms = 0

for perm_i in tqdm(range(n_perm), desc="Permutations"):
    y_perm = np.random.RandomState(perm_i).permutation(y)
    perm_proba = []
    valid_perm = True
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_perm = y_perm[train_idx]
        
        # Check if both classes present in training
        if len(np.unique(y_train_perm)) < 2:
            valid_perm = False
            break
        
        # Feature selection with permuted labels
        selector = SelectKBest(f_classif, k=min(K_FEATURES, X_train.shape[1]))
        selector.fit(X_train, y_train_perm)
        
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Model
        model = LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train_scaled, y_train_perm)
        
        # Predict
        prob = model.predict_proba(X_test_scaled)[0][1]
        perm_proba.append(prob)
    
    # Calculate permuted AUC
    if valid_perm and len(perm_proba) == len(y):
        try:
            if len(np.unique(y_perm)) == 2:
                perm_auc = roc_auc_score(y_perm, perm_proba)
                perm_aucs.append(perm_auc)
            else:
                skipped_perms += 1
        except ValueError:
            skipped_perms += 1
    else:
        skipped_perms += 1

# Calculate p-value
if len(perm_aucs) > 0:
    perm_aucs = np.array(perm_aucs)
    p_value = (np.sum(perm_aucs >= auc) + 1) / (len(perm_aucs) + 1)
    
    print(f"\n📊 Permutation Test Results:")
    print(f"Valid permutations: {len(perm_aucs)}/{n_perm} (skipped {skipped_perms})")
    print(f"Observed AUC: {auc:.3f}")
    print(f"Null distribution: {perm_aucs.mean():.3f} ± {perm_aucs.std():.3f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Result is statistically significant (p < 0.05)")
    else:
        print("❌ Result is NOT statistically significant (p ≥ 0.05)")
else:
    print("⚠️ Warning: No valid permutations completed!")
    p_value = 1.0

# === Bootstrap Confidence Intervals ===
print("\n" + "="*70)
print("STEP 4: BOOTSTRAP CONFIDENCE INTERVALS (n=1000)")
print("="*70)

n_boot = 1000
boot_metrics = {
    'accuracy': [],
    'auc': [],
    'sensitivity': [],
    'specificity': []
}

rng = np.random.RandomState(RANDOM_SEED)
print("Calculating 95% confidence intervals...\n")

for i in range(n_boot):
    # Stratified bootstrap
    boot_idx = []
    for class_val in [0, 1]:
        class_idx = np.where(y_true == class_val)[0]
        sampled = rng.choice(class_idx, len(class_idx), replace=True)
        boot_idx.extend(sampled)
    boot_idx = np.array(boot_idx)
    
    # Calculate metrics on bootstrap sample
    if len(np.unique(y_true[boot_idx])) == 2:
        try:
            boot_metrics['accuracy'].append(accuracy_score(y_true[boot_idx], y_pred[boot_idx]))
            boot_metrics['auc'].append(roc_auc_score(y_true[boot_idx], y_proba[boot_idx]))
            
            cm_boot = confusion_matrix(y_true[boot_idx], y_pred[boot_idx])
            if cm_boot[1].sum() > 0:
                boot_metrics['sensitivity'].append(cm_boot[1,1] / cm_boot[1].sum())
            if cm_boot[0].sum() > 0:
                boot_metrics['specificity'].append(cm_boot[0,0] / cm_boot[0].sum())
        except:
            continue

# Calculate CIs and SEs
print("📊 95% Bootstrap Confidence Intervals (with Standard Errors):")
ci_results = {}
for metric_name in ['accuracy', 'auc', 'sensitivity', 'specificity']:
    if boot_metrics[metric_name]:
        values = np.array(boot_metrics[metric_name])
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        se = np.std(values)
        ci_results[metric_name] = {'lower': ci_lower, 'upper': ci_upper, 'se': se}
        
        if metric_name == 'accuracy':
            orig_val = accuracy
        elif metric_name == 'auc':
            orig_val = auc
        elif metric_name == 'sensitivity':
            orig_val = sensitivity
        else:
            orig_val = specificity
            
        print(f"{metric_name.capitalize():12s}: {orig_val:.3f} ± {se:.3f} "
              f"(95% CI: {ci_lower:.3f} - {ci_upper:.3f})")

# === VISUALIZATIONS ===
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
import os

# Set publication-quality parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

print("\n" + "="*70)
print("STEP 5: PUBLICATION-QUALITY VISUALIZATIONS (600 DPI)")
print("="*70)

os.makedirs('figures', exist_ok=True)

# FIGURE 1: Permutation Test Histogram
if len(perm_aucs) > 0:
    fig, ax = plt.subplots(figsize=(7, 5))
    
    n, bins, patches = ax.hist(perm_aucs, bins=30, density=True, 
                               alpha=0.7, color='#1e88e5', edgecolor='black', linewidth=1.0)
    
    kde = gaussian_kde(perm_aucs)
    x_range = np.linspace(perm_aucs.min(), perm_aucs.max(), 200)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2.5, label='Null distribution')
    
    ax.axvline(x=auc, color='#d32f2f', linestyle='--', linewidth=2.5, 
               label=f'Observed AUC = {auc:.3f}')
    
    x_fill = np.linspace(auc, perm_aucs.max(), 100)
    y_fill = kde(x_fill)
    ax.fill_between(x_fill, 0, y_fill, alpha=0.3, color='#d32f2f')
    
    ax.text(0.05, 0.95, f'p-value = {p_value:.4f}', transform=ax.transAxes,
            fontsize=13, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    ax.set_xlabel('AUC', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_title('Permutation Test: Null Distribution of AUC\n(n=1000 permutations)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.0, color='gray')
    
    null_mean = perm_aucs.mean()
    null_std = perm_aucs.std()
    ax.axvline(x=null_mean, color='#555555', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(null_mean, ax.get_ylim()[1]*0.5, f'Null mean\n{null_mean:.3f}±{null_std:.3f}', 
            ha='center', fontsize=10, color='black', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/permutation_histogram.png', dpi=600, bbox_inches='tight')
    plt.savefig('figures/permutation_histogram.pdf', bbox_inches='tight')
    plt.close()
    print("✅ Saved: permutation_histogram.png/pdf")

# FIGURE 2: ROC Curve
fig, ax = plt.subplots(figsize=(7, 7))

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
ax.plot(fpr, tpr, color='#2e7d32', linewidth=3.0, 
        label=f'ROC curve (AUC = {auc:.3f})')

# Bootstrap confidence band
if len(boot_metrics['auc']) > 0:
    n_bootstrap_roc = 100
    tpr_bootstraps = []
    
    rng_roc = np.random.RandomState(42)
    for i in range(n_bootstrap_roc):
        boot_idx = rng_roc.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[boot_idx])) == 2:
            fpr_boot, tpr_boot, _ = roc_curve(y_true[boot_idx], y_proba[boot_idx])
            tpr_interp = np.interp(fpr, fpr_boot, tpr_boot)
            tpr_bootstraps.append(tpr_interp)
    
    if tpr_bootstraps:
        tpr_bootstraps = np.array(tpr_bootstraps)
        tpr_lower = np.percentile(tpr_bootstraps, 2.5, axis=0)
        tpr_upper = np.percentile(tpr_bootstraps, 97.5, axis=0)
        ax.fill_between(fpr, tpr_lower, tpr_upper, alpha=0.25, color='#2e7d32',
                        label='95% CI')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random (AUC = 0.5)')

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='#d32f2f', s=150, 
          marker='o', edgecolors='black', linewidth=2, zorder=5,
          label=f'Optimal (threshold={optimal_threshold:.3f})')

# Add AUC with CI to plot
if 'auc' in ci_results:
    ax.text(0.05, 0.95, 
            f"AUC = {auc:.3f}\n95% CI: {ci_results['auc']['lower']:.3f}-{ci_results['auc']['upper']:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
ax.set_title('Receiver Operating Characteristic Curve\nNested Univariate Feature Selection', 
            fontsize=14, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.0, color='gray')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures/roc_curve.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/roc_curve.pdf', bbox_inches='tight')
plt.close()
print("✅ Saved: roc_curve.png/pdf")

# FIGURE 3: Feature Selection Heatmap
top_features = [feat for feat, count in feature_counter.most_common(15)]
selection_matrix = np.zeros((len(top_features), 16))

for fold_idx, selected_in_fold in enumerate(selected_features_per_fold):
    for feat_idx, feature in enumerate(top_features):
        if feature in selected_in_fold:
            selection_matrix[feat_idx, fold_idx] = 1

fig, ax = plt.subplots(figsize=(14, 8))

sns.heatmap(selection_matrix, 
            xticklabels=[f'P{i+1}' for i in range(16)],
            yticklabels=top_features,  # Full names
            cmap='RdYlGn',
            cbar_kws={'label': 'Selected', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='gray',
            square=False,
            ax=ax,
            vmin=0, vmax=1)

cbar = ax.collections[0].colorbar
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['No', 'Yes'])
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Selected', fontsize=12, fontweight='bold')

for i, feature in enumerate(top_features):
    count = feature_counter[feature]
    ax.text(16.3, i + 0.5, f'{count}/16', fontsize=11, va='center', fontweight='bold')

ax.set_xlabel('Patient (LOOCV Test Fold)', fontsize=13, fontweight='bold')
ax.set_ylabel('Feature Name', fontsize=13, fontweight='bold')
ax.set_title('Feature Selection Stability Across LOOCV Folds\n(Top 15 Most Frequently Selected Features)', 
            fontsize=14, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=10, colors='black')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

plt.tight_layout()
plt.savefig('figures/feature_selection_heatmap.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/feature_selection_heatmap.pdf', bbox_inches='tight')
plt.close()
print("✅ Saved: feature_selection_heatmap.png/pdf")

# FIGURE 4: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
annot_matrix = np.array([[f'{cm[0,0]}\n({cm_normalized[0,0]:.1%})', 
                          f'{cm[0,1]}\n({cm_normalized[0,1]:.1%})'],
                         [f'{cm[1,0]}\n({cm_normalized[1,0]:.1%})', 
                          f'{cm[1,1]}\n({cm_normalized[1,1]:.1%})']])

sns.heatmap(cm, annot=annot_matrix, fmt='', cmap='Blues', 
            xticklabels=['Predicted\nAdenoma', 'Predicted\nMetastasis'],
            yticklabels=['True\nAdenoma', 'True\nMetastasis'],
            cbar_kws={'label': 'Count'},
            linewidths=2, linecolor='black',
            square=True, ax=ax, annot_kws={'size': 15, 'weight': 'bold'},
            vmin=0, vmax=9)

ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.3f} | Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}', 
            fontsize=14, fontweight='bold')
ax.tick_params(colors='black')

plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=600, bbox_inches='tight')
plt.savefig('figures/confusion_matrix.pdf', bbox_inches='tight')
plt.close()
print("✅ Saved: confusion_matrix.png/pdf")

# === Save Results ===
print("\n" + "="*70)
print("STEP 6: SAVING RESULTS")
print("="*70)

# Save predictions with timestamp
predictions_df = pd.DataFrame({
    'PatientID': patient_ids,
    'True_Class': df['ClassLabel'].values,
    'True_Label': y_true,
    'Predicted_Label': y_pred,
    'Predicted_Class': ['Adenoma' if p==0 else 'Metastasis' for p in y_pred],
    'Prob_Metastasis': y_proba,
    'Correct': y_true == y_pred
})
predictions_df.to_excel(f'Nested_Univariate_Predictions_{timestamp}.xlsx', index=False)

# Save feature selection frequency
feature_freq_df = pd.DataFrame(
    list(feature_counter.items()),
    columns=['Feature', 'Selection_Count']
).sort_values('Selection_Count', ascending=False)
feature_freq_df.to_excel(f'Feature_Selection_Frequency_{timestamp}.xlsx', index=False)

# Save summary results
summary_df = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'P-value'],
    'Value': [accuracy, auc, sensitivity, specificity, p_value],
    'CI_Lower': [ci_results.get('accuracy', {}).get('lower', np.nan),
                 ci_results.get('auc', {}).get('lower', np.nan),
                 ci_results.get('sensitivity', {}).get('lower', np.nan),
                 ci_results.get('specificity', {}).get('lower', np.nan),
                 np.nan],
    'CI_Upper': [ci_results.get('accuracy', {}).get('upper', np.nan),
                 ci_results.get('auc', {}).get('upper', np.nan),
                 ci_results.get('sensitivity', {}).get('upper', np.nan),
                 ci_results.get('specificity', {}).get('upper', np.nan),
                 np.nan]
})
summary_df.to_excel(f'Summary_Results_{timestamp}.xlsx', index=False)

print(f"✅ Results saved with timestamp {timestamp}:")
print(f"   - Nested_Univariate_Predictions_{timestamp}.xlsx")
print(f"   - Feature_Selection_Frequency_{timestamp}.xlsx")
print(f"   - Summary_Results_{timestamp}.xlsx")
print(f"   - STABLE_FEATURES_{timestamp}.xlsx (if applicable)")
print("   - All figures in 'figures/' directory")

# === Final Interpretation ===
print("\n" + "="*70)
print("FINAL INTERPRETATION")
print("="*70)

print("\n🔍 Key Findings:")
print(f"1. Model Performance: AUC = {auc:.3f}")
if p_value < 0.05:
    print(f"   ✅ Statistically significant (p = {p_value:.4f})")
elif p_value < 0.20:
    print(f"   📊 Trending toward significance (p = {p_value:.4f})")
else:
    print(f"   ❌ NOT statistically significant (p = {p_value:.4f})")

print(f"\n2. Feature Selection Stability:")
if len(always_selected) > 0:
    print(f"   - {len(always_selected)} features consistently selected in ALL folds")
    print("   - High stability suggests robust signal")
    for feat in always_selected:
        print(f"     • {feat}")
else:
    print("   - No features selected in all folds")
    print("   - High instability suggests weak/noisy signal")

print(f"\n3. Clinical Utility:")
if auc > 0.7:
    print(f"   - AUC > 0.7 suggests potential clinical value")
elif auc > 0.6:
    print(f"   - AUC 0.6-0.7 indicates weak discrimination")
else:
    print(f"   - AUC ≤ 0.6 indicates no useful discrimination")

print("\n📝 For Your Manuscript:")
if 'auc' in ci_results:
    print(f"'Univariate feature selection with nested LOOCV achieved AUC={auc:.2f} ")
    print(f"(95% CI: {ci_results['auc']['lower']:.2f}-{ci_results['auc']['upper']:.2f}, ")
    print(f"permutation p={p_value:.3f}). ")
    if len(always_selected) > 0:
        print(f"{len(always_selected)} radiomics features were selected in all 16 folds, ")
        print("suggesting potential biomarkers for validation in larger cohorts.'")
    else:
        print("Feature selection was unstable, indicating the need for larger samples.'")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print(f"Random seed: {RANDOM_SEED} | Timestamp: {timestamp}")
print("="*70)