#!/usr/bin/env python3
"""
GeneCure AI - Three Separate Prediction Models
Model 1: Cancer Type Detection (UCI Gene Expression)
Model 2: Patient Survival/Risk Prediction (TCGA PanCan Survival Data)
Model 3: Drug Treatment Recommendation (GDSC)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, 
                            roc_auc_score, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Update these paths to match your data location
# ============================================================================
# MODEL 1: Cancer Detection (UCI Dataset)
UCI_DATA_FILE = Path("/Users/rahul04/Desktop/sam/data/clinical/TCGA-PANCAN-HiSeq-801x20531/data.csv")
UCI_LABELS_FILE = Path("data/clinical/TCGA-PANCAN-HiSeq-801x20531/labels.csv")

# MODEL 2: Survival/Risk Prediction (TCGA PanCan Xena - Survival only)
TCGA_CLINICAL_FILE = Path("data/genomic/Survival_SupplementalTable_S1_20171025_xena_sp")

# MODEL 3: Drug Response (GDSC - Drug response files only)
GDSC1_FILE = Path("data/treatment/GDSC1_fitted_dose_response_27Oct23.xlsx")
GDSC2_FILE = Path("data/treatment/GDSC2_fitted_dose_response_27Oct23.xlsx")


OUTPUT_DIR = Path("./genecure_models")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("GeneCure AI - Multi-Model Training Pipeline")
print("="*80)

# ============================================================================
# MODEL 1: CANCER TYPE DETECTION (EARLY DETECTION)
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: CANCER TYPE DETECTION (Early Detection)")
print("="*80)

if UCI_DATA_FILE.exists() and UCI_LABELS_FILE.exists():
    print("\n[1.1] Loading UCI gene expression data...")
    
    # Load data
    uci_data = pd.read_csv(UCI_DATA_FILE, index_col=0)
    uci_labels = pd.read_csv(UCI_LABELS_FILE)
    
    print(f"   Data shape: {uci_data.shape}")
    print(f"   Labels shape: {uci_labels.shape}")
    
    # Auto-detect sample ID and cancer type columns
    sample_col = None
    for col in uci_labels.columns:
        if 'sample' in col.lower() or 'id' in col.lower() or 'unnamed' in col.lower():
            sample_col = col
            break
    if sample_col is None:
        sample_col = uci_labels.columns[0]
    
    cancer_col = None
    for col in uci_labels.columns:
        if 'class' in col.lower() or 'type' in col.lower() or 'cancer' in col.lower():
            cancer_col = col
            break
    if cancer_col is None:
        cancer_col = uci_labels.columns[-1]
    
    print(f"   Sample ID column: '{sample_col}'")
    print(f"   Cancer type column: '{cancer_col}'")
    
    # Set index and align
    uci_labels = uci_labels.set_index(sample_col)
    common_samples = uci_data.index.intersection(uci_labels.index)
    
    X_uci = uci_data.loc[common_samples]
    y_uci = uci_labels.loc[common_samples, cancer_col]
    
    print(f"   Aligned samples: {len(common_samples)}")
    print(f"   Cancer types: {y_uci.nunique()}")
    print(f"   Distribution:\n{y_uci.value_counts()}")
    
    # Preprocessing
    print("\n[1.2] Preprocessing gene expression data...")
    X_uci = X_uci.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_uci = np.log1p(X_uci)  # Log transform
    
    # Standardize
    scaler_uci = StandardScaler()
    X_uci_scaled = scaler_uci.fit_transform(X_uci)
    
    # PCA
    pca_uci = PCA(n_components=0.95, random_state=42)
    X_uci_pca = pca_uci.fit_transform(X_uci_scaled)
    print(f"   PCA components: {X_uci_pca.shape[1]} (95% variance)")
    
    # Encode labels
    le_uci = LabelEncoder()
    y_uci_encoded = le_uci.fit_transform(y_uci)
    
    # Train-test split
    X_train_uci, X_test_uci, y_train_uci, y_test_uci = train_test_split(
        X_uci_pca, y_uci_encoded, test_size=0.2, random_state=42, stratify=y_uci_encoded
    )
    
    # Train model
    print("\n[1.3] Training cancer detection model...")
    model_cancer_detect = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    model_cancer_detect.fit(X_train_uci, y_train_uci)
    
    # Predictions
    y_pred_uci = model_cancer_detect.predict(X_test_uci)
    y_pred_proba_uci = model_cancer_detect.predict_proba(X_test_uci)
    
    # Metrics
    acc_uci = accuracy_score(y_test_uci, y_pred_uci)
    print(f"\n   ✓ Accuracy: {acc_uci:.4f}")
    
    # Per-class probabilities
    print("\n   Sample Predictions:")
    for i in range(min(5, len(y_test_uci))):
        true_label = le_uci.inverse_transform([y_test_uci[i]])[0]
        pred_label = le_uci.inverse_transform([y_pred_uci[i]])[0]
        confidence = y_pred_proba_uci[i].max()
        print(f"   Sample {i+1}: True={true_label}, Pred={pred_label}, Confidence={confidence:.2%}")
    
    # Save model and preprocessors
    joblib.dump(model_cancer_detect, OUTPUT_DIR / "model1_cancer_detection.pkl")
    joblib.dump(scaler_uci, OUTPUT_DIR / "model1_scaler.pkl")
    joblib.dump(pca_uci, OUTPUT_DIR / "model1_pca.pkl")
    joblib.dump(le_uci, OUTPUT_DIR / "model1_label_encoder.pkl")
    
    print("\n   ✓ Model 1 saved successfully!")
    print(f"      - Cancer detection model")
    print(f"      - Input: Gene expression (any patient)")
    print(f"      - Output: Cancer type + confidence score")
    
else:
    print("\n   ⚠ UCI dataset files not found. Skipping Model 1.")
    print(f"   Expected: {UCI_DATA_FILE} and {UCI_LABELS_FILE}")

# ============================================================================
# MODEL 2: PATIENT SURVIVAL/RISK PREDICTION
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: PATIENT SURVIVAL/RISK PREDICTION")
print("="*80)

if TCGA_CLINICAL_FILE.exists():
    print("\n[2.1] Loading TCGA survival data...")
    
    # Load survival data
    try:
        survival_df = pd.read_csv(TCGA_CLINICAL_FILE, sep='\t', low_memory=False)
    except:
        survival_df = pd.read_csv(TCGA_CLINICAL_FILE, low_memory=False)
    
    print(f"   Survival data shape: {survival_df.shape}")
    
    # Detect sample ID column
    sample_col_surv = None
    for col in survival_df.columns:
        if 'sample' in col.lower() or 'patient' in col.lower() or 'barcode' in col.lower():
            sample_col_surv = col
            break
    if sample_col_surv is None:
        sample_col_surv = survival_df.columns[0]
    
    print(f"   Sample ID column: '{sample_col_surv}'")
    survival_df = survival_df.set_index(sample_col_surv)
    
    # FIXED: Use correct columns for survival analysis
    # Priority: OS.time and OS (Overall Survival)
    time_col = None
    event_col = None
    
    # Look for OS.time, DSS.time, PFI.time, DFI.time
    if 'OS.time' in survival_df.columns:
        time_col = 'OS.time'
        event_col = 'OS'
    elif 'DSS.time' in survival_df.columns:
        time_col = 'DSS.time'
        event_col = 'DSS'
    elif 'PFI.time' in survival_df.columns:
        time_col = 'PFI.time'
        event_col = 'PFI'
    elif 'DFI.time' in survival_df.columns:
        time_col = 'DFI.time'
        event_col = 'DFI'
    else:
        # Fallback to auto-detection
        for col in survival_df.columns:
            col_lower = col.lower()
            if time_col is None and 'time' in col_lower and 'days' not in col_lower:
                time_col = col
            if event_col is None and ('status' in col_lower or 'event' in col_lower):
                event_col = col
    
    print(f"   Time column: '{time_col}'")
    print(f"   Event column: '{event_col}'")
    
    if time_col and event_col and time_col in survival_df.columns and event_col in survival_df.columns:
        # Clean survival data
        survival_clean = survival_df[[time_col, event_col]].copy()
        survival_clean[time_col] = pd.to_numeric(survival_clean[time_col], errors='coerce')
        survival_clean[event_col] = pd.to_numeric(survival_clean[event_col], errors='coerce')
        survival_clean = survival_clean.dropna()
        survival_clean = survival_clean[survival_clean[time_col] > 0]
        
        print(f"   Clean survival samples: {len(survival_clean)}")
        print(f"   Events: {survival_clean[event_col].sum():.0f} ({survival_clean[event_col].mean():.1%})")
        
        if len(survival_clean) > 50:
            # Extract all available clinical/numeric features
            print("\n[2.2] Extracting clinical features...")
            numeric_cols = survival_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in [time_col, event_col]]
            
            categorical_cols = survival_df.select_dtypes(include=['object']).columns.tolist()
            
            print(f"   Numeric features: {len(numeric_cols)}")
            print(f"   Categorical features: {len(categorical_cols)}")
            
            # Build feature matrix
            X_clinical = survival_df.loc[survival_clean.index].copy()
            
            # Process numeric features
            clinical_features = pd.DataFrame(index=survival_clean.index)
            
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                # Only keep columns that have data
                valid_numeric = [c for c in numeric_cols if X_clinical[c].notna().sum() > 0]
                if len(valid_numeric) > 0:
                    numeric_imputed = imputer.fit_transform(X_clinical[valid_numeric])
                    numeric_df = pd.DataFrame(numeric_imputed, index=survival_clean.index, columns=valid_numeric)
                    clinical_features = pd.concat([clinical_features, numeric_df], axis=1)
            
            # Process categorical features (low cardinality only)
            for col in categorical_cols:
                if X_clinical[col].nunique() < 20 and X_clinical[col].notna().sum() > 10:
                    dummies = pd.get_dummies(X_clinical[col], prefix=col, drop_first=True, dummy_na=False)
                    clinical_features = pd.concat([clinical_features, dummies], axis=1)
            
            print(f"   Total features extracted: {clinical_features.shape[1]}")
            
            if clinical_features.shape[1] > 0:
                # Fill any remaining NaN
                clinical_features = clinical_features.fillna(0)
                
                # Standardize
                scaler_clinical = StandardScaler()
                X_clinical_scaled = scaler_clinical.fit_transform(clinical_features)
                
                # Create risk groups based on survival time
                print("\n[2.3] Creating risk groups...")
                survival_median = survival_clean[time_col].median()
                
                # High risk = shorter survival time
                y_risk = (survival_clean[time_col] < survival_median).astype(int)
                
                print(f"   Median survival time: {survival_median:.1f}")
                print(f"   High risk (short survival): {y_risk.sum()} ({y_risk.mean():.1%})")
                print(f"   Low risk (long survival): {(~y_risk.astype(bool)).sum()} ({(~y_risk.astype(bool)).mean():.1%})")
                
                # Train risk classifier
                print("\n[2.4] Training risk classification model...")
                X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
                    X_clinical_scaled, y_risk, test_size=0.2, random_state=42, stratify=y_risk
                )
                
                model_risk = XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
                model_risk.fit(X_train_risk, y_train_risk)
                
                y_pred_risk = model_risk.predict(X_test_risk)
                y_pred_proba_risk = model_risk.predict_proba(X_test_risk)[:, 1]
                
                acc_risk = accuracy_score(y_test_risk, y_pred_risk)
                auc_risk = roc_auc_score(y_test_risk, y_pred_proba_risk)
                
                print(f"   ✓ Risk Classification Accuracy: {acc_risk:.4f}")
                print(f"   ✓ Risk Classification AUC: {auc_risk:.4f}")
                
                # Sample predictions
                print("\n   Sample Risk Predictions:")
                for i in range(min(5, len(y_test_risk))):
                    risk_label = "HIGH RISK" if y_pred_risk[i] == 1 else "LOW RISK"
                    confidence = y_pred_proba_risk[i] if y_pred_risk[i] == 1 else (1 - y_pred_proba_risk[i])
                    print(f"   Patient {i+1}: {risk_label} (confidence: {confidence:.2%})")
                
                # Train Cox Proportional Hazards model
                print("\n[2.5] Training Cox survival model...")
                X_cox = pd.DataFrame(X_clinical_scaled, index=survival_clean.index,
                                    columns=clinical_features.columns)
                X_cox[time_col] = survival_clean[time_col].values
                X_cox[event_col] = survival_clean[event_col].astype(int).values
                
                try:
                    cph = CoxPHFitter(penalizer=0.1)
                    cph.fit(X_cox, duration_col=time_col, event_col=event_col)
                    
                    c_index = concordance_index(
                        X_cox[time_col],
                        -cph.predict_partial_hazard(X_cox),
                        X_cox[event_col]
                    )
                    print(f"   ✓ Cox Model C-index: {c_index:.4f}")
                    
                    # Save models
                    joblib.dump(model_risk, OUTPUT_DIR / "model2_risk_classifier.pkl")
                    joblib.dump(cph, OUTPUT_DIR / "model2_cox_survival.pkl")
                    joblib.dump(scaler_clinical, OUTPUT_DIR / "model2_scaler.pkl")
                    
                    # Save feature names for inference
                    feature_info = {
                        'feature_names': list(clinical_features.columns),
                        'numeric_cols': valid_numeric if len(valid_numeric) > 0 else [],
                        'categorical_cols': [c for c in categorical_cols if X_clinical[c].nunique() < 20]
                    }
                    joblib.dump(feature_info, OUTPUT_DIR / "model2_feature_info.pkl")
                    
                    print("\n   ✓ Model 2 saved successfully!")
                    print(f"      - Risk classifier (High/Low risk groups)")
                    print(f"      - Cox survival model (survival probability)")
                    print(f"      - Input: Clinical features")
                    print(f"      - Output: Risk category + survival prediction")
                    
                except Exception as e:
                    print(f"   ⚠ Cox model training failed: {e}")
                    print("   Saving risk classifier only...")
                    
                    joblib.dump(model_risk, OUTPUT_DIR / "model2_risk_classifier.pkl")
                    joblib.dump(scaler_clinical, OUTPUT_DIR / "model2_scaler.pkl")
                    feature_info = {
                        'feature_names': list(clinical_features.columns),
                        'numeric_cols': valid_numeric if len(valid_numeric) > 0 else [],
                        'categorical_cols': [c for c in categorical_cols if X_clinical[c].nunique() < 20]
                    }
                    joblib.dump(feature_info, OUTPUT_DIR / "model2_feature_info.pkl")
                    
                    print("\n   ✓ Model 2 (risk classifier) saved successfully!")
            else:
                print("   ⚠ No valid features extracted from clinical data")
        else:
            print("   ⚠ Insufficient survival samples after cleaning")
    else:
        print("   ⚠ Could not detect time and event columns")
else:
    print(f"\n   ⚠ TCGA clinical file not found: {TCGA_CLINICAL_FILE}")
    print("   Skipping Model 2.")

# ============================================================================
# MODEL 3: DRUG TREATMENT RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("MODEL 3: DRUG TREATMENT RECOMMENDATION")
print("="*80)

gdsc_files = [GDSC1_FILE, GDSC2_FILE]
gdsc_available = [f for f in gdsc_files if f.exists()]

if gdsc_available:
    print(f"\n[3.1] Loading GDSC drug response data...")
    
    gdsc_dfs = []
    for gdsc_file in gdsc_available:
        try:
            df = pd.read_excel(gdsc_file)
            print(f"   Loaded {gdsc_file.name}: {df.shape}")
            gdsc_dfs.append(df)
        except Exception as e:
            print(f"   ⚠ Error loading {gdsc_file.name}: {e}")
    
    if gdsc_dfs:
        gdsc_combined = pd.concat(gdsc_dfs, ignore_index=True)
        print(f"   Combined GDSC data: {gdsc_combined.shape}")
        print(f"   Available columns: {list(gdsc_combined.columns)}")
        
        # FIXED: Explicitly set correct columns for GDSC data
        cell_line_col = 'CELL_LINE_NAME' if 'CELL_LINE_NAME' in gdsc_combined.columns else None
        drug_col = 'DRUG_NAME' if 'DRUG_NAME' in gdsc_combined.columns else None
        ic50_col = 'LN_IC50' if 'LN_IC50' in gdsc_combined.columns else None
        
        # Fallback if columns don't exist
        if not drug_col:
            for col in gdsc_combined.columns:
                if 'drug' in col.lower() and 'name' in col.lower():
                    drug_col = col
                    break
        
        if not ic50_col:
            for col in gdsc_combined.columns:
                if 'ic50' in col.lower():
                    ic50_col = col
                    break
        
        print(f"\n   Detected columns:")
        print(f"   - Cell line: '{cell_line_col}'")
        print(f"   - Drug: '{drug_col}'")
        print(f"   - IC50/Response: '{ic50_col}'")
        
        if cell_line_col and drug_col and ic50_col:
            # Clean data
            gdsc_clean = gdsc_combined[[cell_line_col, drug_col, ic50_col]].copy()
            gdsc_clean = gdsc_clean.dropna()
            
            print(f"\n   Clean records: {len(gdsc_clean)}")
            print(f"   Unique drugs: {gdsc_clean[drug_col].nunique()}")
            print(f"   Unique cell lines: {gdsc_clean[cell_line_col].nunique()}")
            
            # Get top drugs by number of cell lines tested
            drug_counts = gdsc_clean.groupby(drug_col).size().sort_values(ascending=False)
            top_drugs = drug_counts.head(15)
            
            print(f"\n   Top 15 drugs by number of cell lines tested:")
            for drug, count in top_drugs.items():
                print(f"   - {drug}: {count} cell lines")
            
            # Train drug response models
            print("\n[3.2] Training drug response models...")
            print("   Note: Training without gene expression data.")
            print("   Using cell line patterns for drug response prediction.")
            
            drug_models = {}
            
            for i, drug in enumerate(top_drugs.head(10).index, 1):
                drug_data = gdsc_clean[gdsc_clean[drug_col] == drug].copy()
                
                print(f"\n   [{i}/10] Processing {drug}...")
                print(f"       Samples: {len(drug_data)}")
                
                if len(drug_data) >= 30:  # Lowered threshold
                    # Create features from cell line (one-hot encoding)
                    cell_line_dummies = pd.get_dummies(drug_data[cell_line_col], drop_first=True)
                    
                    # FIXED: Clean column names to remove special characters
                    # XGBoost doesn't allow [, ], < in feature names
                    clean_columns = []
                    for col in cell_line_dummies.columns:
                        clean_col = str(col).replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
                        clean_columns.append(clean_col)
                    cell_line_dummies.columns = clean_columns
                    
                    # Check if we have enough features
                    if cell_line_dummies.shape[1] >= 5:
                        X_drug = cell_line_dummies
                        y_drug = drug_data[ic50_col].values
                        
                        # Check for variance in target
                        if y_drug.std() > 0.01:
                            try:
                                # Train model
                                X_tr, X_te, y_tr, y_te = train_test_split(
                                    X_drug, y_drug, test_size=0.2, random_state=42
                                )
                                
                                model_drug = XGBRegressor(
                                    n_estimators=100,
                                    max_depth=4,
                                    learning_rate=0.1,
                                    random_state=42,
                                    n_jobs=-1,
                                    tree_method='hist'  # More stable for large feature sets
                                )
                                model_drug.fit(X_tr, y_tr)
                                
                                y_pred = model_drug.predict(X_te)
                                r2 = r2_score(y_te, y_pred)
                                mse = mean_squared_error(y_te, y_pred)
                                
                                # Statistics
                                ic50_mean = y_drug.mean()
                                ic50_std = y_drug.std()
                                ic50_min = y_drug.min()
                                ic50_max = y_drug.max()
                                
                                print(f"       ✓ R²={r2:.4f}, MSE={mse:.4f}")
                                print(f"       IC50: μ={ic50_mean:.2f}, σ={ic50_std:.2f}, range=[{ic50_min:.2f}, {ic50_max:.2f}]")
                                
                                # Save model
                                drug_safe_name = drug.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')[:50]
                                joblib.dump(model_drug, OUTPUT_DIR / f"model3_drug_{drug_safe_name}.pkl")
                                
                                # Save cell line encoder info
                                cell_line_info = {
                                    'cell_lines': list(X_drug.columns),
                                    'ic50_mean': float(ic50_mean),
                                    'ic50_std': float(ic50_std),
                                    'ic50_min': float(ic50_min),
                                    'ic50_max': float(ic50_max)
                                }
                                joblib.dump(cell_line_info, OUTPUT_DIR / f"model3_info_{drug_safe_name}.pkl")
                                
                                drug_models[drug] = {
                                    'model': model_drug,
                                    'r2': r2,
                                    'mse': mse,
                                    'n_samples': len(drug_data),
                                    'ic50_mean': ic50_mean,
                                    'ic50_std': ic50_std,
                                    'ic50_min': ic50_min,
                                    'ic50_max': ic50_max
                                }
                            except Exception as e:
                                print(f"       ⚠ Training failed: {e}")
                        else:
                            print(f"       ⚠ Skipped: No variance in IC50 values")
                    else:
                        print(f"       ⚠ Skipped: Too few features ({cell_line_dummies.shape[1]})")
                else:
                    print(f"       ⚠ Skipped: Insufficient samples ({len(drug_data)} < 30)")
            
            if len(drug_models) > 0:
                print(f"\n   ✓ Model 3: Successfully trained {len(drug_models)} drug response models!")
                
                # Save drug list with effectiveness info
                drug_list_df = pd.DataFrame([
                    {
                        'drug': drug,
                        'r2_score': info['r2'],
                        'mse': info['mse'],
                        'samples': info['n_samples'],
                        'mean_ic50': info['ic50_mean'],
                        'std_ic50': info['ic50_std'],
                        'min_ic50': info['ic50_min'],
                        'max_ic50': info['ic50_max']
                    }
                    for drug, info in drug_models.items()
                ])
                drug_list_df = drug_list_df.sort_values('r2_score', ascending=False)
                drug_list_df.to_csv(OUTPUT_DIR / "model3_drug_list.csv", index=False)
                
                print(f"\n      📊 Model 3 Summary:")
                print(f"      - Trained models for {len(drug_models)} drugs")
                print(f"      - Input: Cell line characteristics")
                print(f"      - Output: Predicted IC50 (lower = more effective)")
                
                # Create drug ranking system
                print("\n[3.3] Creating drug effectiveness ranking...")
                
                # Lower IC50 = more effective
                drug_effectiveness = []
                for drug, info in drug_models.items():
                    # Effectiveness score: inverse of mean IC50
                    effectiveness_score = 1 / (1 + info['ic50_mean'])
                    
                    # Categorize based on IC50 thresholds (LN_IC50 scale)
                    if info['ic50_mean'] < 0:  # Log scale: negative = more potent
                        category = 'Highly Effective'
                    elif info['ic50_mean'] < 1:
                        category = 'Moderately Effective'
                    else:
                        category = 'Less Effective'
                    
                    drug_effectiveness.append({
                        'drug': drug,
                        'effectiveness_score': effectiveness_score,
                        'mean_ic50': info['ic50_mean'],
                        'category': category,
                        'model_r2': info['r2']
                    })
                
                drug_ranking_df = pd.DataFrame(drug_effectiveness)
                drug_ranking_df = drug_ranking_df.sort_values('mean_ic50', ascending=True)  # Lower IC50 = better
                drug_ranking_df.to_csv(OUTPUT_DIR / "model3_drug_ranking.csv", index=False)
                
                print("   ✓ Drug ranking system created!")
                print(f"\n   🏆 Top 5 Most Effective Drugs (by IC50):")
                for idx, row in enumerate(drug_ranking_df.head(5).iterrows(), 1):
                    _, data = row
                    print(f"   {idx}. {data['drug']}")
                    print(f"      Category: {data['category']}")
                    print(f"      Mean IC50: {data['mean_ic50']:.3f} (lower is better)")
                    print(f"      Model R²: {data['model_r2']:.3f}")
                
            else:
                print("\n   ⚠ No drug models trained successfully")
                print("   Possible reasons:")
                print("   - Insufficient data per drug")
                print("   - Low variance in IC50 values")
                print("   - Not enough cell line diversity")
        else:
            print("   ⚠ Could not detect required columns in GDSC data")
            print(f"   Need: cell_line_col={cell_line_col}, drug_col={drug_col}, ic50_col={ic50_col}")
else:
    print(f"\n   ⚠ GDSC files not found")
    print(f"   Expected: {GDSC1_FILE} or {GDSC2_FILE}")
    print("   Skipping Model 3.")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAINING COMPLETE - GENECURE AI SYSTEM")
print("="*80)
print(f"\n📁 Models saved to: {OUTPUT_DIR.absolute()}\n")

print("🎯 THREE PREDICTION MODELS TRAINED:\n")

print("1️⃣  CANCER DETECTION MODEL (Early Detection)")
print("    Purpose: Detect cancer type from gene expression")
print("    Input: Patient gene expression profile")
print("    Output: Cancer type + confidence score")
print("    Example: '94% chance of breast cancer subtype A'\n")

print("2️⃣  SURVIVAL/RISK PREDICTION MODEL")
print("    Purpose: Classify patients into risk groups")
print("    Input: Clinical features (stage, age, tumor size, etc.)")
print("    Output: HIGH RISK or LOW RISK + survival probability")
print("    Example: 'HIGH RISK patient with 35% 5-year survival'\n")

print("3️⃣  DRUG TREATMENT RECOMMENDATION MODEL")
print("    Purpose: Predict which drugs work best")
print("    Input: Cell line / patient characteristics")
print("    Output: Drug effectiveness ranking")
print("    Example: 'Top drug: Cisplatin (Highly Effective)'\n")

print("📊 USAGE:")
print("   - Load models: joblib.load('model_path.pkl')")
print("   - For new patients:")
print("     * Model 1: Predict cancer type from genes")
print("     * Model 2: Assess survival risk from clinical data")
print("     * Model 3: Recommend best treatment drugs")

print("\n" + "="*80)
print("GeneCure AI Training Pipeline Complete!")
print("="*80)