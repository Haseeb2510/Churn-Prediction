import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import scipy.stats as stats
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, 'data')

def load_data():
    data_file = os.path.join(data_path, 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(data_file)
    return df

def comprehensive_data_diagnosis(df):
    """Run comprehensive diagnostics on the dataset"""
    
    print("🔍 COMPREHENSIVE DATA DIAGNOSIS")
    print("=" * 50)
    
    # 1. Basic info
    print("\n1. DATASET SHAPE:")
    print(f"   Samples: {df.shape[0]}, Features: {df.shape[1]}")
    
    # 2. Target distribution
    print("\n2. TARGET DISTRIBUTION:")
    target_counts = df['Churn'].value_counts()
    print(f"   No Risk (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
    print(f"   Risk (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")
    
    # 3. Check for missing values
    print("\n3. MISSING VALUES:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("   ✅ No missing values")
    else:
        print("   ❌ Missing values found:")
        for col, count in missing.items():
            print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
    
    # 4. Check for constant features
    print("\n4. CONSTANT FEATURES:")
    constant_features = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_features.append(col)
    if constant_features:
        print(f"   ❌ Constant features: {constant_features}")
    else:
        print("   ✅ No constant features")
    
    # 5. Check correlation with target
    print("\n5. FEATURE CORRELATION WITH TARGET:")
    correlations = df.corr()['Churn'].abs().sort_values(ascending=False)
    
    # Remove target itself
    correlations = correlations[1:]
    
    print("   Top 10 correlations:")
    for feature, corr in correlations.head(10).items():
        print(f"      {feature}: {corr:.4f}")
    
    print("   Bottom 10 correlations:")
    for feature, corr in correlations.tail(10).items():
        print(f"      {feature}: {corr:.4f}")
    
    # 6. Check for highly correlated features
    print("\n6. FEATURE CORRELATION MATRIX ANALYSIS:")
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    
    for col in upper_tri.columns:
        high_corr = upper_tri[col][upper_tri[col] > 0.8]
        for feature, corr in high_corr.items():
            high_corr_pairs.append((col, feature, corr))
    
    if high_corr_pairs:
        print("   ❌ Highly correlated features (>0.8):")
        for pair in high_corr_pairs[:5]:  # Show top 5
            print(f"      {pair[0]} vs {pair[1]}: {pair[2]:.4f}")
    else:
        print("   ✅ No highly correlated feature pairs")
    
    # 7. Mutual information with target
    print("\n7. MUTUAL INFORMATION WITH TARGET:")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    print("   Top 10 mutual information scores:")
    for feature, score in mi_series.head(10).items():
        print(f"      {feature}: {score:.4f}")
    
    # 8. Feature distributions by target class
    print("\n8. FEATURE DISTRIBUTION ANALYSIS:")
    print("   Features with significant difference between classes (t-test p-value < 0.05):")
    
    significant_features = []
    for col in X.columns:
        group1 = df[df['Churn'] == 0][col]
        group2 = df[df['Churn'] == 1][col]
        
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        if p_value < 0.05: # type: ignore
            significant_features.append((col, p_value, abs(group1.mean() - group2.mean())))
    
    # Sort by effect size (mean difference)
    significant_features.sort(key=lambda x: x[2], reverse=True)
    
    for feature, p_val, effect_size in significant_features[:10]:
        print(f"      {feature}: p={p_val:.6f}, effect={effect_size:.4f}")
    
    return correlations, mi_series, significant_features

def create_simplified_model(df, top_features=15):
    """Create a model using only the most predictive features"""
    
    # Get feature importance using Random Forest
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    
    # Get feature importance
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 RANDOM FOREST FEATURE IMPORTANCE:")
    print(feature_imp.head(20))
    
    # Select top features
    selected_features = feature_imp.head(top_features)['feature'].tolist()
    print(f"\n📋 SELECTED TOP {top_features} FEATURES:")
    print(selected_features)
    
    return selected_features, feature_imp

def plot_diagnostic_charts(df, correlations, mi_series, significant_features):
    """Create diagnostic plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation with target
    top_corr = correlations.head(10)
    axes[0, 0].barh(range(len(top_corr)), top_corr.values)
    axes[0, 0].set_yticks(range(len(top_corr)))
    axes[0, 0].set_yticklabels(top_corr.index)
    axes[0, 0].set_title('Top 10 Features Correlated with Target')
    axes[0, 0].set_xlabel('Absolute Correlation')
    
    # 2. Mutual information
    top_mi = mi_series.head(10)
    axes[0, 1].barh(range(len(top_mi)), top_mi.values)
    axes[0, 1].set_yticks(range(len(top_mi)))
    axes[0, 1].set_yticklabels(top_mi.index)
    axes[0, 1].set_title('Top 10 Features by Mutual Information')
    axes[0, 1].set_xlabel('Mutual Information')
    
    # 3. Feature importance from Random Forest
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1, 0].barh(range(len(feature_imp)), feature_imp['importance'])
    axes[1, 0].set_yticks(range(len(feature_imp)))
    axes[1, 0].set_yticklabels(feature_imp['feature'])
    axes[1, 0].set_title('Top 10 Features by Random Forest Importance')
    axes[1, 0].set_xlabel('Importance')
    
    # 4. Target distribution
    target_counts = df['Churn'].value_counts()
    axes[1, 1].pie(target_counts.values, labels=['No Risk', 'Risk'], autopct='%1.1f%%')
    axes[1, 1].set_title('Target Variable Distribution')
    
    plt.tight_layout()
    plt.show()

# Run the comprehensive diagnosis
print("Loading data...")
df = load_data()

print("Running comprehensive diagnosis...")
correlations, mi_scores, significant_features = comprehensive_data_diagnosis(df)

print("\nCreating diagnostic charts...")
plot_diagnostic_charts(df, correlations, mi_scores, significant_features)

print("\nCreating simplified feature set...")
selected_features, feature_imp = create_simplified_model(df)

print("\n" + "="*50)
print("RECOMMENDED NEXT STEPS:")
print("1. Review the diagnostic output above")
print("2. Train model using only top features:", selected_features[:10])
print("3. Consider if there are data quality issues")
print("4. Verify feature engineering logic")
print("="*50)
