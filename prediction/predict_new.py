import os, joblib
import xgboost as xgb
from xgboost import XGBClassifier
import tensorflow as tf
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(project_root, 'data/new')

class ChurnPredictor:
    def __init__(self, model_path, metadata_path, model_type='auto'):
        # Load metadata first
        self.metadata = joblib.load(metadata_path)
        self.threshold = self.metadata['threshold']
        self.model_type = model_type

        # Auto-detect model type from file extension
        if model_type == 'auto':
            self.model_type = self._detect_model_type_from_path(model_path)

        # Load the appropriate model
        self.model =self._load_model(model_path)

        print(f"✅ {self.model_type.upper()} Predictor loaded successfully")
        print(f"🎯 Using threshold: {self.threshold:.3f}")

    def _detect_model_type_from_path(self, model_path):
        # Detect model type from file extension
        model_ext = model_path.lower().split('.')[-1]

        if model_ext in ['json', 'joblib', 'pkl', 'model']:
            return 'xgboost'
        elif model_ext in ['h5', 'keras', 'tf']:
            return 'tensorflow'
        else:
            # Default to XGBoost for backward compatibility
            print(f"⚠️  Unknown model extension '{model_ext}', defaulting to XGBoost")
            return 'xgboost'

    def _load_model(self, model_path):
        if self.model_type == 'xgboost':
            model = XGBClassifier() 
            model.load_model(model_path)
            return model

        elif self.model_type == 'tensorflow':
            try:
                # Try loading as keras model
                model = tf.keras.models.load_model(model_path) # type:ignore
                return model
            except Exception as e:
                print(f"❌ Error loading Tensorflow model: {e}")
                raise
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess_new_data(self, new_data):
        """Prepare new data for testing"""
        def load_encoders():
            model_dir = os.path.join(project_root, 'model')
            encoders_path = os.path.join(model_dir, 'categorical_encoders.joblib')

            if not os.path.exists(encoders_path):
                raise FileNotFoundError(f"Encoder file not found at {encoders_path}. Train model.")

            return joblib.load(encoders_path)

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
            
            if 'churn' in df_eng.columns:
                cols = [c for c in df_eng.columns if c != 'Churn']
                df_eng = df_eng[cols + ['Churn']] 

            return df_eng

        def engineered_data_processing(df: pd.DataFrame):
            
            target = 'Churn'

            # Handle missing values in numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            # Convert Churn to binary
            if target in df.columns:
                df[target] = df[target].map({'Yes': 1, 'No': 0}).astype(np.int32)

            # Identify categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            categorical_cols = [c for c in categorical_cols if c != target]

            # Encode categorical features
            encoders = load_encoders()
            for col in categorical_cols:
                le = encoders[col]
                # Check for unseern categories
                unseen_categories = set(df[col].astype(str)) - set(le.classes_)
                if unseen_categories:
                    print(f"⚠️ Replacing unseen categories in {col}: {unseen_categories}")
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))

            # Ensure consistent numeric types
            df = df.astype({col: np.float32 for col in numeric_cols if col != target})
            if target in df.columns:
                df[target] = df[target].astype(np.int32)

            # Handle potential NaNs from inf conversions
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Column arrangement (target last)
            if target in df.columns:
                cols = [c for c in df.columns if c != target]
                df = df[cols + [target]]

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

        def prepare_data_for_trainig(df, numeric_cols):
            
            models_dir = os.path.join(project_root, 'model')

            target = 'Churn'
            if target in df.columns:
                X = df.drop(columns=[target])
            else:
                X = df.copy()

            def load_scaler():
                scaler_path = os.path.join(models_dir, 'scaler.joblib')
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"Scaler not found at {scaler_path}")
                
                scaler = joblib.load(scaler_path)
                
                # Verify scaler is fitted
                if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                    raise ValueError("Scaler is not fitted. Please retrain the model.")
                
                print(f"✅ Loaded scaler with {len(scaler.mean_)} features")
                return scaler

            scaler =load_scaler()

            # Ensure we only scale columns that exist in both data and scaler
            available_numeric_cols = [col for col in numeric_cols if col in X.columns]
            print(f"📊 Scaling {len(available_numeric_cols)} numeric columns")

            # Only transform existing numeric columns
            X[available_numeric_cols] = scaler.transform(X[available_numeric_cols])


            return X

        df = create_engineered_features(new_data)
        df, numeric_cols = engineered_data_processing(df)
        X = prepare_data_for_trainig(df, numeric_cols)

        return X

    def predict_data(self, data):
        """Get probability predictions"""
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(data)[:, 1]
            return probabilities
        elif hasattr(self.model, 'predict'):
            raw_pred = self.model.predict(data, verbose=0)
            
            if len(raw_pred.shape) == 1:
                return raw_pred
            elif raw_pred.shape[1] == 1:
                return raw_pred.flatten()
            else:
                return raw_pred[:, 1]
        else:
            raise ValueError("Model doesn't support probability predictions")
    
    def predict(self, data, custom_threshold=None):
        """Make churn predictions using saved threshold"""
        threshold = custom_threshold if custom_threshold is not None else self.threshold

        probabilities = self.predict_data(data)
        predictions = (probabilities > threshold).astype(int)

        return predictions, probabilities

    def predict_with_confidence(self, data):
        """Get predictions with confidence scores"""
        probabilities  = self.predict_data(data)
        predictions = (probabilities  > self.threshold).astype(int) 
        
        # Calculate confidence (distance from threshold) 
        confidence = np.abs(probabilities  - self.threshold)   

        results = []
        for i , (pred, prob, conf) in enumerate(zip(predictions, probabilities , confidence)):
            results.append({
                'client_id': i,
                'churn_prediction': pred,
                'churn_probability': prob,
                'confidence': conf,
                'risk_level': self._get_risk_level(prob, self.threshold)
            })

        return pd.DataFrame(results)
    
    def _get_risk_level(self, probability, threshold):
        """Categorize risk levels based on probability"""
        if probability < threshold - 0.2:
            return "Low Risk"
        elif probability < threshold:
            return 'Medium Risk'
        elif probability < threshold + 0.2:
            return 'High Risk'
        else:
            return "Very High Risk"


class Ensemble:
    def __init__(self, xgb_predictor, nn_predictor, ensemble_method='precision_focused'):
        self.xgb_predictor = xgb_predictor
        self.nn_predictor = nn_predictor
        self.ensemble_method = ensemble_method
        
        # Store individual model predictions
        self.xgb_probas = None
        self.nn_probas = None
        self.printed = False

    def predict_proba(self, data):
        # Get individual model predictions
        if not self.printed:
            self.xgb_probas = self.xgb_predictor.predict_data(data)
            self.nn_probas = self.nn_predictor.predict_data(data)
            self.printed = True

            print(f"📈 Model Probability Ranges:")
            print(f"   XGBoost - Min: {self.xgb_probas.min():.3f}, Max: {self.xgb_probas.max():.3f}")
            print(f"   Neural Net - Min: {self.nn_probas.min():.3f}, Max: {self.nn_probas.max():.3f}")
    
        if self.ensemble_method == 'precision_focused':
            return self._precision_focused()
        elif self.ensemble_method == 'recall_focused':
            return self._recall_focused()
        elif self.ensemble_method == 'balanced':
            return self._balanced_ensemble()
        elif self.ensemble_method == 'threshold_adaptive':
            return self._threshold_adaptive()
        elif self.ensemble_method == 'conservative':
            return self._conservative_ensemble()
        else:
            return self._balanced_ensemble()

    def _precision_focused(self):
        """Favor NN's precison - requires both models to agree for positive prediction"""
        final_proba = np.zeros_like(self.xgb_probas)

        for i, (xgb_p, nn_p) in enumerate(zip(self.xgb_probas, self.nn_probas)): # type: ignore
            # Only predict churn if both models are reasonanbly confident
            if xgb_p > 0.5 and nn_p  > 0.4: # NN has lower thresold since it's more conservative
                final_proba[i] = max(xgb_p, nn_p)
            elif xgb_p > 0.7: # Very high confidence frem XGB
                final_proba[i] = xgb_p * 0.7 # Discount solo XGB predicions
            else:
                final_proba[i] = (xgb_p * 0.3 + nn_p * 0.7) # Favor NN for low-risk

        return final_proba 

    def _recall_focused(self):
        """Favor XGB'S recall - predict churn if either model suggests it"""
        final_proba = np.zeros_like(self.xgb_probas)

        for i, (xgb_p, nn_p) in enumerate(zip(self.xgb_probas, self.nn_probas)): # type: ignore
            # Predict churn if either model is confident
            if xgb_p > 0.4 or nn_p > 0.6:
                final_proba[i] = max(xgb_p, nn_p)
            else:
                final_proba[i] = min(xgb_p, nn_p)
        
        return final_proba
    
    def _balanced_ensemble(self):
        """Try to balance precision and recall"""
        final_proba = np.zeros_like(self.xgb_probas)

        for i, (xgb_p, nn_p) in enumerate(zip(self.xgb_probas, self.nn_probas)): # type: ignore
            # Use NN's prediction when it's confident, otherwise use weighted average
            if nn_p > 0.7: # NN very confident about churn
                final_proba[i] = nn_p
            elif nn_p < 0.3: # Very confident about no churn
                final_proba[i] = nn_p
            else: # Middle ground - weighted average favoring NN fro precision
                final_proba[i] = (xgb_p * 0.4 + nn_p * 0.6)

        return final_proba

    def _conservative_ensemble(self):
        """Very conservative - reduce false positions at the cost of some recall"""
        final_proba = np.zeros_like(self.xgb_probas)

        for i, (xgb_p, nn_p) in enumerate(zip(self.xgb_probas, self.nn_probas)): # type: ignore
            # Only predict churn with high confidence from both
            if xgb_p > 0.6 and nn_p > 0.5:
                final_proba[i] = (xgb_p + nn_p) / 2
            elif xgb_p > 0.8: # Extremely high confidence from XGB
                final_proba[i] = xgb_p * 0.8
            else:
                final_proba[i] =  min(xgb_p, nn_p)
        
        return final_proba
    
    def _threshold_adaptive(self):
        """Use diffrent strategies based on probability ranges"""
        final_proba = np.zeros_like(self.xgb_probas)

        for i, (xgb_p, nn_p) in enumerate(zip(self.xgb_probas, self.nn_probas)): # type: ignore
            # Low risk zone - favor NN (more conservative)
            if xgb_p < 0.3 and nn_p < 0.3:
                final_proba[i] = min(xgb_p, nn_p)
            # High risk zone - require agreement
            elif xgb_p > 0.7 and nn_p > 0.5:
                final_proba[i] = (xgb_p + nn_p) / 2
            # Medium risk zone - weighted average
            else:
                # Give more weight to the more confident model
                xgb_conf = abs(xgb_p - 0.5)
                nn_conf = abs(nn_p - 0.5)
                total_conf = xgb_conf + nn_conf

                if total_conf > 0:
                    final_proba[i] = (xgb_p * xgb_conf + nn_p * nn_conf) / total_conf
                else:
                    final_proba[i] = (xgb_p + nn_p) / 2
        return final_proba

    def predict_with_confidence(self, data, threshold=0.55):
        ensemble_proba = self.predict_proba(data) 
        ensemble_proba = np.array(ensemble_proba)
        predictions =(ensemble_proba > threshold).astype(int) 
        confidence = np.abs(ensemble_proba - threshold)

        results = []
        for i, (pred, prob, conf) in enumerate(zip(predictions, ensemble_proba, confidence)):
            results.append({
                'client_id': i,
                'churn_prediction': pred,
                'churn_probability': prob,
                'confidence': conf,
                'risk_level': self._get_risk_level(prob, threshold),
                'model_source': 'ensemble'
            })  

        return pd.DataFrame(results)

    def _get_risk_level(self, probability, threshold):
        """Categorize risk levels based on probability"""
        if probability < threshold - 0.2:
            return "Low Risk"
        elif probability < threshold:
            return 'Medium Risk'
        elif probability < threshold + 0.2:
            return 'High Risk'
        else:
            return "Very High Risk"



def load_data():
    # Use this method to load new customers data
    data_file = os.path.join(data_path, 'customers.csv') # This data isn't very reliable it's ai generated
    df = pd.read_csv(data_file)
    return df

def load_xgb_model():
    model_dir = os.path.join(project_root, 'model/xgb')
    
    # Find model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith('xgboost_balanced') and f.endswith('.json')]
    metadata_files = [f for f in os.listdir(model_dir) if f.startswith('xgboost_balanced_metadata_')]

    if not model_files or not metadata_files:
        raise FileNotFoundError("No model files found. Train a model first")

    latest_model = sorted(model_files)[-1]
    latest_metadata = sorted(metadata_files)[-1]

    model_path = os.path.join(model_dir, latest_model)
    metadata_path = os.path.join(model_dir, latest_metadata)

    # Initialize predictor
    predictor = ChurnPredictor(model_path, metadata_path, 'xgboost')
    return predictor

def load_nn_model():
    model_dir = os.path.join(project_root, 'model/nn')
    
    # Find model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith('nn_model') and f.endswith('.h5')]
    metadata_files = [f for f in os.listdir(model_dir) if f.startswith('meta_data')]

    # Add more descriptive error messages
    if not model_files:
        raise FileNotFoundError(f"No .h5 model files found in {model_dir}")
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {model_dir}")

    latest_model = sorted(model_files)[-1]
    latest_metadata = sorted(metadata_files)[-1]

    model_path = os.path.join(model_dir, latest_model)
    metadata_path = os.path.join(model_dir, latest_metadata)

    # Initialize predictor
    predictor = ChurnPredictor(model_path, metadata_path)
    return predictor

def create_ensemble(xgb_model, nn_model, method):
    ensemble = Ensemble(xgb_model, nn_model, ensemble_method=method)
    return ensemble


def predict_new_clients(raw_df):
    predictor = load_xgb_model()

    worked_df = predictor.preprocess_new_data(raw_df)

    # Make predictions
    print("🔮 Makin predictions......")
    results = predictor.predict_with_confidence(worked_df)

    # Display results
    print("\n📊 PREDICTION RESULTS:")
    print("=" * 50)
    for _, row in results.iterrows():
        status = "🚨 CHURN RISK" if row['churn_prediction'] == 1 else "✅ LOW RISK"
        print(f'Client {row['client_id']}: {status}')
        print(f"    Probability: {row['churn_probability']:.3f} | Confidence: {row['confidence']:.3f}")
        print(f"    Risk Level: {row['risk_level']}")
        print("-" * 40)
    

    return results

def predict_with_comparison(raw_df, predictor, model, t=False):
    """Make predictions and compare with actual Churn values"""
    # Preprocess data (this should handle the Churn column properly)
    if t: # If we pass an ensemble
        xgb_mdoel = load_xgb_model()
        processed_data = xgb_mdoel.preprocess_new_data(raw_df)

    else:
        processed_data = predictor.preprocess_new_data(raw_df)
    
    # Make predictions
    print(f"🔮 Making predictions with {model} model...")
    results = predictor.predict_with_confidence(processed_data)
    
    # Add actual Churn values to results
    results = add_actual_churn_comparison(results, raw_df)
    
    # Display comparison results
    # display_comparison_results(results)
    
    # Calculate and display performance metrics
    calculate_performance_metrics(results)
    
    return results

def add_actual_churn_comparison(results_df, original_df):
    """Add actual Churn values to the results for comparison"""
    
    # Extract actual Churn values (handle both 'Yes'/'No' and 1/0 formats)
    if 'Churn' in original_df.columns:
        actual_churn = original_df['Churn'].copy()
        
        # Convert to binary if needed
        if actual_churn.dtype == 'object':
            actual_churn = actual_churn.map({'Yes': 1, 'No': 0})
        
        # Add to results
        results_df['actual_churn'] = actual_churn.values
        results_df['correct_prediction'] = results_df['churn_prediction'] == results_df['actual_churn']
        
        # Add prediction status
        results_df['prediction_status'] = results_df.apply(
            lambda row: get_prediction_status(row['churn_prediction'], row['actual_churn']), 
            axis=1
        )
    
    return results_df

def get_prediction_status(predicted, actual):
    """Get human-readable prediction status"""
    if predicted == 1 and actual == 1:
        return "✅ True Positive (Correctly predicted churn)"
    elif predicted == 0 and actual == 0:
        return "✅ True Negative (Correctly predicted no churn)"
    elif predicted == 1 and actual == 0:
        return "❌ False Positive (False alarm)"
    elif predicted == 0 and actual == 1:
        return "❌ False Negative (Missed churn)"
    else:
        return "Unknown"

def display_comparison_results(results_df):
    """Display detailed comparison between predictions and actual values"""
    
    print("\n" + "="*80)
    print("📊 PREDICTION vs ACTUAL COMPARISON")
    print("="*80)
    
    for _, row in results_df.iterrows():
        status = "🚨 CHURN RISK" if row['churn_prediction'] == 1 else "✅ LOW RISK"
        actual_status = "Churned" if row['actual_churn'] == 1 else "Not Churned"
        
        print(f"Client {row['client_id']}:")
        print(f"   Prediction: {status}")
        print(f"   Actual:     {actual_status}")
        print(f"   Probability: {row['churn_probability']:.3f} | Confidence: {row['confidence']:.3f}")
        print(f"   Risk Level: {row['risk_level']}")
        print(f"   Status: {row['prediction_status']}")
        
        # Add emoji based on correctness
        if row['correct_prediction']:
            print("   🎯 ACCURATE PREDICTION!")
        else:
            print("   ⚠️  MISMATCH!")
        
        print("-" * 60)

def calculate_performance_metrics(results_df):
    """Calculate and display performance metrics"""
    
    if 'actual_churn' not in results_df.columns:
        print("❌ No actual Churn data available for comparison")
        return
    
    # Calculate metrics
    total = len(results_df)
    correct = results_df['correct_prediction'].sum()
    accuracy = correct / total
    
    # Confusion matrix components
    true_positive = ((results_df['churn_prediction'] == 1) & (results_df['actual_churn'] == 1)).sum()
    true_negative = ((results_df['churn_prediction'] == 0) & (results_df['actual_churn'] == 0)).sum()
    false_positive = ((results_df['churn_prediction'] == 1) & (results_df['actual_churn'] == 0)).sum()
    false_negative = ((results_df['churn_prediction'] == 0) & (results_df['actual_churn'] == 1)).sum()
    
    # Additional metrics
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("📈 PERFORMANCE METRICS")
    print("="*60)
    print(f"Total Customers: {total}")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total} correct)")
    print(f"Precision: {precision:.1%} ({true_positive}/{true_positive + false_positive} churn predictions correct)")
    print(f"Recall: {recall:.1%} ({true_positive}/{true_positive + false_negative} actual churners caught)")
    print(f"F1-Score: {f1_score:.3f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"True Positives:  {true_positive:3d} (Correctly predicted churn)")
    print(f"True Negatives:  {true_negative:3d} (Correctly predicted no churn)")
    print(f"False Positives: {false_positive:3d} (False alarms)")
    print(f"False Negatives: {false_negative:3d} (Missed churners)")
    
    # Business impact
    print(f"\n💼 BUSINESS IMPACT:")
    print(f"Churners identified: {true_positive}/{true_positive + false_negative}")
    print(f"False alarms: {false_positive}")
    print(f"Missed opportunities: {false_negative}")

def test_improved_ensembles():
    df = load_data()
    xgb_model = load_xgb_model()
    nn_model = load_nn_model()

    original_data = pd.read_csv(os.path.join(project_root, 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'))
    processed_data = nn_model.preprocess_new_data(original_data)

    methods = ['precision_focused', 'recall_focused', 'balanced', 'conservative', 'threshold_adaptive']

    print("🤖 INDIVIDUAL MODEL PERFORMANCE:")
    print("XGBoost: Precision=59.8%, Recall=79.5%")
    print("Neural Net: Precision=64.0%, Recall=64.9%")
    print("\n" + "="*70)
    
    for method in methods:
        print(f"\n🎯 TESTING {method.upper()} ENSEMBLE")
        print("=" * 50)

        ensemble = Ensemble(xgb_model, nn_model, ensemble_method=method)

        # Test multiple threshold to find optimal balance
        best_f1 = 0
        best_threshold = 0.5

        for threshold in [0.45, 0.5, 0.55, 0.6, 0.65]:
            results = ensemble.predict_with_confidence(processed_data, threshold=threshold)
            results = add_actual_churn_comparison(results, original_data)
            
            # Calculate F1 score manually
            tp = ((results['churn_prediction'] == 1) & (results['actual_churn'] == 1)).sum()
            fp = ((results['churn_prediction'] == 1) & (results['actual_churn'] == 0)).sum()
            fn = ((results['churn_prediction'] == 0) & (results['actual_churn'] == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"🏆 Best threshold for {method}: {best_threshold} (F1: {best_f1:.3f})")
        
        # Show results with best threshold
        results = ensemble.predict_with_confidence(processed_data, threshold=best_threshold)
        results = add_actual_churn_comparison(results, original_data)
        calculate_performance_metrics(results)

# demo.py    

if __name__ == "__main__":
    test_improved_ensembles()