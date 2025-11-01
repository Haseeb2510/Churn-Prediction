import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, 'data')

def load_data():
    data_file = os.path.join(data_path, 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = pd.read_csv(data_file)
    return df

def save_worked_data(df):
    data_file = os.path.join(data_path, 'worked', 'worked_enginered_WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df.to_csv(data_file, index=False)

def save_encoders(encoders):
    model_dir = os.path.join(project_root, 'model')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(encoders, os.path.join(model_dir, 'categorical_encoders.joblib'))

def create_engineered_features(df: pd.DataFrame):
    df_eng = df.copy()

    df_eng['TotalCharges'] = pd.to_numeric(df_eng['TotalCharges'], errors='coerce')
    df_eng = df_eng.fillna(0)

    # 1. CUSTOMER LIFETIME VALUE
    df_eng['AvgMonthlySpend'] = df_eng['TotalCharges'] / df_eng['tenure'].replace(0, 1)
    df_eng['AvgMonthlySpend'] = df_eng['AvgMonthlySpend'].replace([np.inf, -np.inf], 0).fillna(0)

    df_eng['CustomerValueScore'] = df_eng['MonthlyCharges'] * df_eng['tenure']


    # 2. SERVICE USAGE
    premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_eng['NumPremiumServices'] = df_eng[premium_services].apply(lambda x: (x == 'Yes').sum(), axis=1)

    df_eng['HasFiberOptic'] = (df_eng['InternetService'] == 'Fiber optic').astype(int) 
    df_eng['HasDSL'] = (df_eng['InternetService'] == 'DSL').astype(int) 


    # 3. CONTRACT & PAYMENT BEHAVIOUR
    payment_behaviour = {'Month-to-month': 1, 'One ear': 12, 'Two year': 24}
    df_eng['ContractMonth'] = df_eng['Contract'].map(payment_behaviour)

    df_eng['AutoPayment'] = df_eng['PaymentMethod'].str.contains('automatic').astype(int)
    df_eng['PaperLessAuto'] = ((df_eng['PaperlessBilling'] == 'Yes') & 
                            (df_eng['AutoPayment'] == 1)).astype(int)
    
    # 4. TENURE-BASED FEATURES
    df_eng['TenureGroup'] = pd.cut(df_eng['tenure'], 
                                 bins=[0, 12, 24, 36, 48, 60, np.inf],
                                 labels=['New', 'Growing', 'Established', 'Mature', 'Loyal', 'Veteran'])

    df_eng['IsNewCustomer'] = (df_eng['tenure'] <= 12).astype(int)
    df_eng['IsAtRiskPeriod'] = ((df_eng['tenure'] > 12) & (df_eng['tenure'] <= 24)).astype(int)

    # 5. SERVICE BUNDLE
    conditions = [
        (df_eng['InternetService'] == 'No') & (df_eng['PhoneService'] == 'No'),
        (df_eng['InternetService'] == 'No') & (df_eng['PhoneService'] == 'Yes'),
        (df_eng['InternetService'] == 'DSL') & (df_eng['PhoneService'] == 'No'),
        (df_eng['InternetService'] == 'DSL') & (df_eng['PhoneService'] == 'Yes'),
        (df_eng['InternetService'] == 'Fiber optic') & (df_eng['PhoneService'] == 'No'),
        (df_eng['InternetService'] == 'Fiber optic') & (df_eng['PhoneService'] == 'Yes')
    ]
    choices = ['Basic', 'PhoneOnly', 'DSLOnly', 'DSLBundle', 'FiberOnly', 'FiberBundle']
    df_eng['ServiceBundle'] = np.select(conditions, choices, default='Unknown')


    # 6. FINANCIAL RATIOS
    df_eng['SpendToTenureRatio'] = df_eng['MonthlyCharges'] / (df_eng['tenure'].replace(0, 1))
    df_eng['ValueRetentionScore'] = df_eng['TotalCharges'] / (df_eng['MonthlyCharges'] +1)


    # 7. DEMOGRAFIC INTERACTIONS
    df_eng['SeniorWithDependents'] = ((df_eng['SeniorCitizen'] == 1) & (df_eng['Dependents'] == 'Yes')).astype(int)
    df_eng['PartnerNoDependents'] = ((df_eng['Partner'] == 'Yes') & (df_eng['Dependents'] == 'No')).astype(int)

    # 8. BEHAVIORAL FLAGS
    df_eng['HighValueNew'] = ((df_eng['MonthlyCharges'] > df_eng['MonthlyCharges'].quantile(0.75)) & (df_eng['tenure'] < 6)).astype(int)
    df_eng['ContracEndingRisk'] = ((df_eng['Contract'] == 'Month-to-month') | (df_eng['tenure']) % 12 == 11).astype(int)


    # 9. PEER GROUP COMPARISONS
    df_eng['SpendvsPeer'] = df_eng['MonthlyCharges'] / df_eng.groupby('ServiceBundle')['MonthlyCharges'].transform('mean')
    df_eng['TenureVsPeer'] = df_eng['tenure'] / df_eng['tenure'].mean()


    # 10. TREND INDICATORS
    df_eng['PricePerService'] = df_eng['MonthlyCharges'] / (df_eng['NumPremiumServices'].replace(0, 1))
    max_services = 6
    df_eng['ServiceUtilizationRate'] = df_eng['NumPremiumServices'] / max_services


    # 11. RISK SCORE
    risk_factors = [
        (df_eng['PaperlessBilling'] == 'Yes'),
        (df_eng['Contract'] == 'Month-to-month'),
        (df_eng['PaymentMethod'] == 'Electronic check'),
        (df_eng['tenure'] < 12 ),
        (df_eng['NumPremiumServices'] == 0)
    ]

    df_eng['RiskFactorCount'] = sum(condition.astype(int) for condition in risk_factors)


    # 1. Domain-Specific Features
    df_eng['TenureGroup_SpendInteraction'] = df_eng['tenure'] * df_eng['MonthlyCharges'] / 100
    df_eng['ContractRiskScore'] = ((df_eng['Contract'] == 'Month-to-month') * 2 + 
                            (df_eng['PaperlessBilling'] == 'Yes') * 1)

    # 2. Behavioral Patterns
    df_eng['SpendIncreaseRecently'] = (df_eng['MonthlyCharges'] > df_eng['TotalCharges'] / df_eng['tenure']).astype(int)

    # 3. Customer Segmentation
    conditions = [
        (df_eng['tenure'] < 12) & (df_eng['MonthlyCharges'] > 70),
        (df_eng['tenure'] > 24) & (df_eng['NumPremiumServices'] < 2),
        (df_eng['Contract'] == 'Month-to-month') & (df_eng['SeniorCitizen'] == 1)
    ]
    choices = ['HighRiskNew', 'LowEngagementLoyal', 'SeniorFlexible']
    df_eng['CustomerSegment'] = np.select(conditions, choices, default='Standard')


    # DROP USLESS COLUMNS
    cols_to_drop = [
        'customerID', 'AvgMonthlySpend', 'CustomerValueScore', 
        'ContractMonth', 'TenureGroup', 'IsNewCustomer', 
        'IsAtRiskPeriod', 'HasDSL', 'HasFiberOptic', 
        'PaperlessBilling', 'TenureVsPeer'
    ]

    df_eng.drop(columns=[c for c in cols_to_drop if c in df_eng.columns], inplace=True)

    cols = [c for c in df_eng.columns if c != 'Churn']
    df_eng = df_eng[cols + ['Churn']] 

    print(df_eng.info())
    for col in df_eng.columns:
        print(col, df_eng[col].unique())
    return df_eng

def engineered_data_processing(df: pd.DataFrame):
    print("=== Basic Data Overview ===")
    print(df.info())
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nUnique value counts per column:\n", df.nunique())

    target = 'Churn'

    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Convert Churn to binary
    df[target] = df[target].map({'Yes': 1, 'No': 0}).astype(np.int32)

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != target]

    print("\nCategorical columns:", categorical_cols)

    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    save_encoders(encoders)  

    # Ensure consistent numeric types
    df = df.astype({col: np.float32 for col in numeric_cols if col != target})
    df[target] = df[target].astype(np.int32)

    # Handle potential NaNs from inf conversions
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Column arrangement (target last)
    cols = [c for c in df.columns if c != target]
    df = df[cols + [target]]

    # Summary statistics
    print("\n=== Cleaned Data Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df[target].value_counts(normalize=True)}")
    print(f"Data types:\n{df.dtypes.value_counts()}")

    print("\nSample of cleaned data:\n", df.head())
    print(df.info())

    numeric_cols = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'SpendToTenureRatio',
        'ValueRetentionScore',
        'SpendvsPeer',
        'PricePerService',
        'ServiceUtilizationRate',
        'NumPremiumServices', 
        'RiskFactorCount',
        'TenureGroup_SpendInteraction'
    ]
    return df, numeric_cols

def debug_scaling(X_train, X_test, numeric_cols):
    """Check scaling results"""
    print("🔍 SCALING VERIFICATION:")
    print(f"Original numeric columns: {numeric_cols}")
    
    # Check a few scaled values
    sample_idx = 0
    print(f"\nSample row {sample_idx} scaled values:")
    for i, col in enumerate(numeric_cols):
        if i < 5:  # Show first 5 numeric features
            print(f"  {col}: {X_train[sample_idx, i]:.4f}")
    
    # Check overall statistics
    print(f"\nScaled data statistics:")
    print(f"  Mean: {np.mean(X_train[:, :len(numeric_cols)]):.4f}")
    print(f"  Std: {np.std(X_train[:, :len(numeric_cols)]):.4f}")
    print(f"  Range: [{np.min(X_train[:, :len(numeric_cols)]):.4f}, {np.max(X_train[:, :len(numeric_cols)]):.4f}]")

def prepare_data_for_trainig(df, numeric_cols):
    
    models_dir = os.path.join(project_root, 'model')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def save_scaler(scaler):
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    save_scaler(scaler)

    X_train = X_train.values.astype(np.float32)
    X_test  = X_test.values.astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    y_test  = y_test.values.astype(np.float32)
    
    debug_scaling(X_train, X_test, numeric_cols)

    datasets = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train, 
        "y_test": y_test
    }

    for name, data in datasets.items():
        joblib.dump(data, os.path.join(models_dir, f"{name}.joblib"))

if __name__ == '__main__':
    df = load_data()
    df = create_engineered_features(df)
    df, numeric_cols = engineered_data_processing(df)
    save_worked_data(df)
    prepare_data_for_trainig(df, numeric_cols)
