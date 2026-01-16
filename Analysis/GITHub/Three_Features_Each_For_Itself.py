import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

df = pd.read_excel("aggregated_radiomics_features.xlsx")
y = df['ClassLabel'].map({'Adenoma': 0, 'Metastasis': 1}).values

# Test each of the 3 stable features
features_to_test = {
    'Sphericity': 'original-shape-Sphericity',
    'wavelet-LHL-glcm-Idn': 'wavelet-LHL-glcm-Idn',
    'wavelet-LLL-glrlm': 'wavelet-LLL-glrlm-LowGrayLevelRunEmphasis'
}

for name, feature in features_to_test.items():
    X = df[[feature]]
    loo = LeaveOneOut()
    y_proba = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train_scaled, y_train)
        y_proba.append(model.predict_proba(X_test_scaled)[0][1])
    
    auc = roc_auc_score(y, y_proba)
    
    # Permutation test for p-value
    perm_aucs = []
    for i in tqdm(range(1000), desc=f"Permutation for {name}"):
        y_perm = np.random.RandomState(i).permutation(y)
        perm_proba = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_perm = y_perm[train_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(class_weight='balanced', random_state=42)
            model.fit(X_train_scaled, y_train_perm)
            perm_proba.append(model.predict_proba(X_test_scaled)[0][1])
        
        perm_aucs.append(roc_auc_score(y_perm, perm_proba))
    
    p_value = (np.sum(np.array(perm_aucs) >= auc) + 1) / 1001
    print(f"\n{name}:")
    print(f"  AUC: {auc:.3f}")
    print(f"  P-value: {p_value:.4f}")