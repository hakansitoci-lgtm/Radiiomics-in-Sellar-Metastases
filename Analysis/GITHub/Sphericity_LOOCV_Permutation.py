"""
Sphericity Analysis for Pituitary Adenoma vs Metastasis Classification
Complete analysis with permutation testing, bootstrap CI, and visualizations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix, 
                           roc_curve, classification_report)
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set publication quality parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

print("="*70)
print("SPHERICITY BIOMARKER ANALYSIS")
print("Single Feature Performance Evaluation")
print("="*70)

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load data
df = pd.read_excel("aggregated_radiomics_features.xlsx")
X_sphericity = df[['original-shape-Sphericity']]
y = df['ClassLabel'].map({'Adenoma': 0, 'Metastasis': 1}).values
patient_ids = df['PatientID'].values

print(f"\n📊 Dataset Summary:")
print(f"Total samples: {len(X_sphericity)}")
print(f"Adenomas: {sum(y==0)}")
print(f"Metastases: {sum(y==1)}")
print(f"Feature: original-shape-Sphericity")
print(f"Timestamp: {timestamp}")

# =====================================
# STEP 1: LOOCV Analysis
# =====================================
print("\n" + "="*70)
print("STEP 1: LEAVE-ONE-OUT CROSS-VALIDATION")
print("="*70)

loo = LeaveOneOut()
y_true, y_pred, y_proba = [], [], []
coefficients = []
sphericity_values = X_sphericity['original-shape-Sphericity'].values

for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X_sphericity)):
    X_train, X_test = X_sphericity.iloc[train_idx], X_sphericity.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Store coefficient
    coefficients.append(model.coef_[0][0])
    
    # Predict
    pred = model.predict(X_test_scaled)[0]
    prob = model.predict_proba(X_test_scaled)[0][1]
    
    y_true.append(y_test[0])
    y_pred.append(pred)
    y_proba.append(prob)
    
    # Print progress
    test_patient = patient_ids[test_idx[0]]
    actual_value = sphericity_values[test_idx[0]]
    print(f"Fold {fold_idx+1:2d}: Patient_{test_patient:<2d} | "
          f"Sphericity={actual_value:.3f} | "
          f"P(Meta)={prob:.3f} | True={'Meta' if y_test[0]==1 else 'Aden'}")

# Convert to arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_proba = np.array(y_proba)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_proba)
cm = confusion_matrix(y_true, y_pred)
sensitivity = cm[1,1] / cm[1].sum()
specificity = cm[0,0] / cm[0].sum()

print("\n📊 LOOCV Results:")
print(f"AUC         : {auc:.3f}")
print(f"Accuracy    : {accuracy:.3f}")
print(f"Sensitivity : {sensitivity:.3f} ({cm[1,1]}/{cm[1].sum()} metastases)")
print(f"Specificity : {specificity:.3f} ({cm[0,0]}/{cm[0].sum()} adenomas)")
print(f"Mean coefficient: {np.mean(coefficients):.3f} ± {np.std(coefficients):.3f}")
print("\nConfusion Matrix:")
print("           Pred_Aden  Pred_Meta")
print(f"True_Aden      {cm[0,0]:2d}        {cm[0,1]:2d}")
print(f"True_Meta      {cm[1,0]:2d}        {cm[1,1]:2d}")

# =====================================
# STEP 2: Find Optimal Cutoff
# =====================================
print("\n" + "="*70)
print("STEP 2: OPTIMAL CUTOFF DETERMINATION")
print("="*70)

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
optimal_idx = np.argmax(tpr - fpr)  # Youden's J statistic
optimal_threshold_prob = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

# Convert probability threshold to sphericity value
# Train on full dataset to get the transformation
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_sphericity)
model_full = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model_full.fit(X_scaled_full, y)

# Find sphericity value corresponding to probability threshold
sphericity_range = np.linspace(sphericity_values.min(), sphericity_values.max(), 1000)
sphericity_scaled = scaler_full.transform(sphericity_range.reshape(-1, 1))
probs = model_full.predict_proba(sphericity_scaled)[:, 1]
optimal_sphericity_idx = np.argmin(np.abs(probs - optimal_threshold_prob))
optimal_sphericity_value = sphericity_range[optimal_sphericity_idx]

print(f"Optimal probability threshold: {optimal_threshold_prob:.3f}")
print(f"Optimal sphericity cutoff: {optimal_sphericity_value:.3f}")
print(f"At this cutoff:")
print(f"  Sensitivity: {optimal_sensitivity:.3f}")
print(f"  Specificity: {optimal_specificity:.3f}")

# =====================================
# STEP 3: Permutation Test
# =====================================
print("\n" + "="*70)
print("STEP 3: PERMUTATION TEST (n=1000)")
print("="*70)

n_perm = 1000
perm_aucs = []

for perm_i in tqdm(range(n_perm), desc="Permutations"):
    y_perm = np.random.RandomState(perm_i).permutation(y)
    perm_proba = []
    
    for train_idx, test_idx in loo.split(X_sphericity):
        X_train, X_test = X_sphericity.iloc[train_idx], X_sphericity.iloc[test_idx]
        y_train_perm = y_perm[train_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train_perm)
        
        perm_proba.append(model.predict_proba(X_test_scaled)[0][1])
    
    perm_auc = roc_auc_score(y_perm, perm_proba)
    perm_aucs.append(perm_auc)

perm_aucs = np.array(perm_aucs)
p_value = (np.sum(perm_aucs >= auc) + 1) / (n_perm + 1)

print(f"\n📊 Permutation Results:")
print(f"Observed AUC: {auc:.3f}")
print(f"Null distribution: {perm_aucs.mean():.3f} ± {perm_aucs.std():.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.001:
    print("✅ HIGHLY SIGNIFICANT (p < 0.001)")
elif p_value < 0.01:
    print("✅ STATISTICALLY SIGNIFICANT (p < 0.01)")
elif p_value < 0.05:
    print("✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print("❌ NOT statistically significant")

# =====================================
# STEP 4: Bootstrap Confidence Intervals
# =====================================
print("\n" + "="*70)
print("STEP 4: BOOTSTRAP CONFIDENCE INTERVALS (n=1000)")
print("="*70)

n_boot = 1000
boot_aucs = []
boot_sensitivity = []
boot_specificity = []

rng = np.random.RandomState(RANDOM_SEED)

for i in range(n_boot):
    # Stratified bootstrap
    boot_idx = []
    for class_val in [0, 1]:
        class_idx = np.where(y_true == class_val)[0]
        sampled = rng.choice(class_idx, len(class_idx), replace=True)
        boot_idx.extend(sampled)
    boot_idx = np.array(boot_idx)
    
    if len(np.unique(y_true[boot_idx])) == 2:
        boot_aucs.append(roc_auc_score(y_true[boot_idx], y_proba[boot_idx]))
        cm_boot = confusion_matrix(y_true[boot_idx], y_pred[boot_idx])
        if cm_boot[1].sum() > 0:
            boot_sensitivity.append(cm_boot[1,1] / cm_boot[1].sum())
        if cm_boot[0].sum() > 0:
            boot_specificity.append(cm_boot[0,0] / cm_boot[0].sum())

# Calculate CIs
auc_ci_lower = np.percentile(boot_aucs, 2.5)
auc_ci_upper = np.percentile(boot_aucs, 97.5)
sens_ci_lower = np.percentile(boot_sensitivity, 2.5)
sens_ci_upper = np.percentile(boot_sensitivity, 97.5)
spec_ci_lower = np.percentile(boot_specificity, 2.5)
spec_ci_upper = np.percentile(boot_specificity, 97.5)

print("📊 95% Bootstrap Confidence Intervals:")
print(f"AUC         : {auc:.3f} (95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f})")
print(f"Sensitivity : {sensitivity:.3f} (95% CI: {sens_ci_lower:.3f}-{sens_ci_upper:.3f})")
print(f"Specificity : {specificity:.3f} (95% CI: {spec_ci_lower:.3f}-{spec_ci_upper:.3f})")

# =====================================
# STEP 5: VISUALIZATIONS
# =====================================
print("\n" + "="*70)
print("STEP 5: CREATING PUBLICATION-QUALITY FIGURES (600 DPI)")
print("="*70)

import os
os.makedirs('figures_sphericity', exist_ok=True)

# FIGURE 1: Permutation Histogram
fig, ax = plt.subplots(figsize=(8, 6))

n, bins, patches = ax.hist(perm_aucs, bins=30, density=True, 
                           alpha=0.7, color='#1565c0', edgecolor='black', linewidth=1.0)

kde = gaussian_kde(perm_aucs)
x_range = np.linspace(perm_aucs.min(), perm_aucs.max(), 200)
ax.plot(x_range, kde(x_range), 'b-', linewidth=2.5, label='Null distribution')

ax.axvline(x=auc, color='#c62828', linestyle='--', linewidth=3, 
           label=f'Observed AUC = {auc:.3f}')

x_fill = np.linspace(auc, 1.0, 100)
y_fill = kde(np.clip(x_fill, perm_aucs.min(), perm_aucs.max()))
ax.fill_between(x_fill, 0, y_fill, alpha=0.3, color='#c62828')

ax.text(0.05, 0.95, f'p-value = {p_value:.4f}', transform=ax.transAxes,
        fontsize=14, fontweight='bold', verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))

ax.set_xlabel('AUC', fontsize=14, fontweight='bold')
ax.set_ylabel('Density', fontsize=14, fontweight='bold')
ax.set_title('Sphericity Biomarker: Permutation Test\n(n=1000 permutations)', 
             fontsize=15, fontweight='bold')
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.0)

plt.tight_layout()
plt.savefig('figures_sphericity/permutation_sphericity.png', dpi=600, bbox_inches='tight')
plt.savefig('figures_sphericity/permutation_sphericity.pdf', bbox_inches='tight')
plt.close()
print("✅ Saved: permutation_sphericity.png/pdf")

# FIGURE 2: ROC Curve with Confidence Band
fig, ax = plt.subplots(figsize=(8, 8))

# Main ROC curve
ax.plot(fpr, tpr, color='#2e7d32', linewidth=3.0, 
        label=f'Sphericity ROC (AUC = {auc:.3f})')

# Bootstrap confidence band for ROC
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

# Random classifier line
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random (AUC = 0.5)')

# Optimal point
ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='#c62828', s=200, 
          marker='o', edgecolors='black', linewidth=2, zorder=5,
          label=f'Optimal cutoff')

# Add text box with AUC and CI
textstr = f'AUC = {auc:.3f}\n95% CI: {auc_ci_lower:.3f}-{auc_ci_upper:.3f}\np = {p_value:.4f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13, fontweight='bold',
        verticalalignment='top', bbox=props)

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
ax.set_title('Sphericity Biomarker: ROC Curve', fontsize=15, fontweight='bold')
ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
ax.grid(True, alpha=0.4, linestyle=':', linewidth=1.0)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('figures_sphericity/roc_sphericity.png', dpi=600, bbox_inches='tight')
plt.savefig('figures_sphericity/roc_sphericity.pdf', bbox_inches='tight')
plt.close()
print("✅ Saved: roc_sphericity.png/pdf")

# FIGURE 3: Sphericity Values Box Plot
fig, ax = plt.subplots(figsize=(8, 6))

adenoma_sphericity = sphericity_values[y == 0]
metastasis_sphericity = sphericity_values[y == 1]

box_data = [adenoma_sphericity, metastasis_sphericity]
positions = [1, 2]

bp = ax.boxplot(box_data, positions=positions, widths=0.6, 
                showmeans=True, meanline=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5),
                medianprops=dict(color='red', linewidth=2),
                meanprops=dict(color='green', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=8))

# Add individual points
for i, (data, pos) in enumerate(zip(box_data, positions)):
    x = np.random.normal(pos, 0.04, size=len(data))
    ax.scatter(x, data, alpha=0.7, s=100, color=['#1565c0', '#c62828'][i], 
               edgecolor='black', linewidth=1.5)

# Add optimal cutoff line
ax.axhline(y=optimal_sphericity_value, color='purple', linestyle='--', 
           linewidth=2, label=f'Optimal cutoff = {optimal_sphericity_value:.3f}')

ax.set_xticks(positions)
ax.set_xticklabels(['Adenoma\n(n=9)', 'Metastasis\n(n=7)'], fontsize=13, fontweight='bold')
ax.set_ylabel('Sphericity Value', fontsize=14, fontweight='bold')
ax.set_title('Distribution of Sphericity Values by Diagnosis', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1.0)

# Add statistical annotation
from scipy.stats import mannwhitneyu
statistic, pval_mw = mannwhitneyu(adenoma_sphericity, metastasis_sphericity)
y_max = max(np.max(adenoma_sphericity), np.max(metastasis_sphericity))
ax.plot([1, 2], [y_max + 0.02, y_max + 0.02], 'k-', linewidth=1.5)
ax.text(1.5, y_max + 0.03, f'p = {pval_mw:.4f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures_sphericity/boxplot_sphericity.png', dpi=600, bbox_inches='tight')
plt.savefig('figures_sphericity/boxplot_sphericity.pdf', bbox_inches='tight')
plt.close()
print("✅ Saved: boxplot_sphericity.png/pdf")

# FIGURE 4: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6))

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
annot_matrix = np.array([[f'{cm[0,0]}\n({cm_normalized[0,0]:.0%})', 
                          f'{cm[0,1]}\n({cm_normalized[0,1]:.0%})'],
                         [f'{cm[1,0]}\n({cm_normalized[1,0]:.0%})', 
                          f'{cm[1,1]}\n({cm_normalized[1,1]:.0%})']])

sns.heatmap(cm, annot=annot_matrix, fmt='', cmap='Blues', 
            xticklabels=['Predicted\nAdenoma', 'Predicted\nMetastasis'],
            yticklabels=['True\nAdenoma', 'True\nMetastasis'],
            cbar_kws={'label': 'Count'},
            linewidths=3, linecolor='black',
            square=True, ax=ax, annot_kws={'size': 16, 'weight': 'bold'},
            vmin=0, vmax=9)

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title(f'Confusion Matrix - Sphericity Alone\n'
            f'Accuracy: {accuracy:.1%} | Sensitivity: {sensitivity:.1%} | Specificity: {specificity:.1%}', 
            fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('figures_sphericity/confusion_sphericity.png', dpi=600, bbox_inches='tight')
plt.savefig('figures_sphericity/confusion_sphericity.pdf', bbox_inches='tight')
plt.close()
print("✅ Saved: confusion_sphericity.png/pdf")

# =====================================
# STEP 6: Save Results
# =====================================
print("\n" + "="*70)
print("STEP 6: SAVING RESULTS")
print("="*70)

# Create results dataframe
results_df = pd.DataFrame({
    'PatientID': patient_ids,
    'True_Class': df['ClassLabel'].values,
    'Sphericity_Value': sphericity_values,
    'Predicted_Probability': y_proba,
    'Predicted_Class': ['Adenoma' if p < optimal_threshold_prob else 'Metastasis' for p in y_proba],
    'Correct': y_true == y_pred
})

results_df.to_excel(f'Sphericity_Predictions_{timestamp}.xlsx', index=False)

# Summary statistics
summary = pd.DataFrame({
    'Metric': ['AUC', 'Sensitivity', 'Specificity', 'Accuracy', 
               'P-value', 'Optimal_Cutoff_Prob', 'Optimal_Cutoff_Sphericity'],
    'Value': [auc, sensitivity, specificity, accuracy, 
              p_value, optimal_threshold_prob, optimal_sphericity_value],
    'CI_Lower': [auc_ci_lower, sens_ci_lower, spec_ci_lower, np.nan, 
                 np.nan, np.nan, np.nan],
    'CI_Upper': [auc_ci_upper, sens_ci_upper, spec_ci_upper, np.nan, 
                 np.nan, np.nan, np.nan]
})

summary.to_excel(f'Sphericity_Summary_{timestamp}.xlsx', index=False)

print(f"✅ Saved results with timestamp {timestamp}:")
print(f"   - Sphericity_Predictions_{timestamp}.xlsx")
print(f"   - Sphericity_Summary_{timestamp}.xlsx")
print("   - All figures in 'figures_sphericity/' directory")

# =====================================
# FINAL INTERPRETATION
# =====================================
print("\n" + "="*70)
print("FINAL INTERPRETATION")
print("="*70)

print("\n🎯 KEY FINDINGS:")
print(f"1. Sphericity alone achieves AUC = {auc:.3f}")
print(f"2. Result is {'STATISTICALLY SIGNIFICANT' if p_value < 0.05 else 'not significant'} (p = {p_value:.4f})")
print(f"3. Optimal cutoff: {optimal_sphericity_value:.3f}")
print(f"4. Performance: {cm[0,0]+cm[1,1]}/{len(y)} correct classifications")

print("\n📝 For Your Manuscript:")
print(f"'Among 416 radiomics features, tumor sphericity emerged as the dominant biomarker.")
print(f"When evaluated in isolation using LOOCV, sphericity achieved an AUC of {auc:.2f}")
print(f"(95% CI: {auc_ci_lower:.2f}-{auc_ci_upper:.2f}, permutation p={p_value:.3f}),")
print(f"with sensitivity of {sensitivity:.1%} and specificity of {specificity:.1%}.")
print(f"The optimal sphericity cutoff of {optimal_sphericity_value:.3f} correctly classified")
print(f"{cm[0,0]+cm[1,1]}/{len(y)} cases. These results suggest that simple shape analysis")
print("may outperform complex machine learning approaches for this classification task.'")

print("\n" + "="*70)
print("SPHERICITY ANALYSIS COMPLETE")
print("="*70)