import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel("aggregated_radiomics_features.xlsx")

# Get all feature columns (excluding identifiers)
feature_cols = [col for col in df.columns if col not in ['ClassLabel', 'PatientID']]
print(f"✅ Total features to analyze: {len(feature_cols)}")

# Define stable features
three_stable = [
    'original-shape-Sphericity',
    'wavelet-LHL-glcm-Idn',
    'wavelet-LLL-glrlm-LowGrayLevelRunEmphasis'
]

# ✅ Check if stable features exist in the dataset
print(f"\n🎯 Checking if stable features are in the dataset...")
for f in three_stable:
    if f in feature_cols:
        print(f"   ✅ {f}")
    else:
        print(f"   ❌ {f} - NOT FOUND!")

# (Optional) Top 10 ANOVA features (expand as needed)
top_10_anova = [
    'original-shape-Sphericity',
    'wavelet-LHL-glcm-Idn',
    'wavelet-LLL-glrlm-LowGrayLevelRunEmphasis',
    # Add the other 7 features here...
]

# Compute correlation matrix (absolute Pearson)
corr_matrix = df[feature_cols].corr().abs()

# === Function to remove highly correlated features ===
def remove_highly_correlated(corr_matrix, threshold=0.9):
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    kept = [col for col in corr_matrix.columns if col not in to_drop]
    return to_drop, kept

# === CORRELATION FILTERING ANALYSIS ===
print("\n" + "="*70)
print("📉 CORRELATION FILTERING ANALYSIS (with stable feature tracking)")
print("="*70)

thresholds = [0.7, 0.8, 0.9, 0.95]
for thresh in thresholds:
    dropped, kept = remove_highly_correlated(corr_matrix, thresh)
    
    print(f"\n--- Threshold: |r| > {thresh} ---")
    print(f"Removed: {len(dropped)} feature(s)")
    print(f"Remaining: {len(kept)} feature(s)")

    survived = [f for f in three_stable if f in kept]
    removed = [f for f in three_stable if f in dropped]

    if survived:
        print("✅ Stable features retained:")
        for f in survived:
            print(f"   - {f}")
    if removed:
        print("❌ Stable features removed:")
        for f in removed:
            high_corr_partners = [
                (other, corr_matrix.loc[f, other])
                for other in feature_cols if other != f and corr_matrix.loc[f, other] > thresh
            ]
            high_corr_partners.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"   - {f} (e.g., r = {high_corr_partners[0][1]:.3f} with {high_corr_partners[0][0]})")

# === DETAILED CORRELATION ANALYSIS OF STABLE FEATURES ===
print("\n" + "="*70)
print("🔍 DETAILED CORRELATION ANALYSIS OF THREE STABLE FEATURES")
print("="*70)

for feature in three_stable:
    high_corr = corr_matrix[feature].drop(labels=[feature])
    high_corr = high_corr[high_corr > 0.5].sort_values(ascending=False)
    
    print(f"\n📌 {feature}")
    if high_corr.empty:
        print("   ✅ No features with |r| > 0.5")
    else:
        print("   🔗 Top correlated features (|r| > 0.5):")
        for idx, val in high_corr.head(5).items():
            print(f"     - {idx}: r = {val:.3f}")

# === CORRELATIONS AMONG THREE STABLE FEATURES ===
print("\n" + "="*70)
print("🔗 CORRELATIONS BETWEEN THE THREE STABLE FEATURES")
print("="*70)

stable_corr = df[three_stable].corr().round(3)
print(stable_corr)

# === FLATTENED CORRELATION PAIRS ===
upper_tri_all = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
flat_corrs = upper_tri_all.values.flatten()
flat_corrs = flat_corrs[~np.isnan(flat_corrs)]

# === SUMMARY STATISTICS OF ALL CORRELATIONS ===
print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)
print(f"Total feature pairs: {len(flat_corrs)}")
print(f"Pairs with |r| > 0.7: {sum(flat_corrs > 0.7)} ({100 * sum(flat_corrs > 0.7) / len(flat_corrs):.1f}%)")
print(f"Pairs with |r| > 0.8: {sum(flat_corrs > 0.8)} ({100 * sum(flat_corrs > 0.8) / len(flat_corrs):.1f}%)")
print(f"Pairs with |r| > 0.9: {sum(flat_corrs > 0.9)} ({100 * sum(flat_corrs > 0.9) / len(flat_corrs):.1f}%)")

# === VISUALIZATION ===
plt.figure(figsize=(12, 5))

# Histogram of all pairwise correlations
plt.subplot(1, 2, 1)
plt.hist(flat_corrs, bins=50, edgecolor='black', alpha=0.75)
plt.axvline(x=0.7, color='orange', linestyle='--', label='r = 0.7')
plt.axvline(x=0.8, color='red', linestyle='--', label='r = 0.8')
plt.axvline(x=0.9, color='darkred', linestyle='--', label='r = 0.9')
plt.title("Distribution of Absolute Feature Correlations")
plt.xlabel("Correlation Coefficient (|r|)")
plt.ylabel("Frequency")
plt.legend()

# Individual correlation distributions for stable features
plt.subplot(1, 2, 2)
for i, feature in enumerate(three_stable):
    fcorr = corr_matrix[feature].drop(labels=[feature])
    plt.hist(fcorr, bins=30, alpha=0.5, label=f"Feature {i+1}: {feature.split('-')[-1]}")
    
plt.title("Stable Feature Correlation Distributions")
plt.xlabel("Absolute Correlation")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("correlation_analysis.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Correlation analysis complete. See 'correlation_analysis.png' for figures.")
