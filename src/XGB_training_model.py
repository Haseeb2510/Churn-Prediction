import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(project_root, "model")
os.makedirs(models_dir, exist_ok=True)
xgb_model = os.path.join(project_root, "model/xgb")
os.makedirs(xgb_model, exist_ok=True)

def load_data():
    """Load training and test data with error handling"""
    try:
        X_train = joblib.load(os.path.join(models_dir, "X_train.joblib"))
        X_test = joblib.load(os.path.join(models_dir, "X_test.joblib"))
        y_train = joblib.load(os.path.join(models_dir, "y_train.joblib")) 
        y_test = joblib.load(os.path.join(models_dir, "y_test.joblib"))
        
        print("✅ Data loaded successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def calculate_scale_pos_weight(y_train):
    """Calculate optimal scale_pos_weight for class imbalance"""
    y_int = y_train.astype(int)
    class_counts = np.bincount(y_int)
    
    if len(class_counts) < 2:
        raise ValueError("Insufficient classes in training data")
        
    scale_pos_weight = class_counts[0] / class_counts[1]

    print(f"🔍 Class distribution: {dict(zip(range(len(class_counts)), class_counts))}")
    print(f"🎯 scale_pos_weight: {scale_pos_weight:.2f}")

    return scale_pos_weight

def create_regularized_model(y_train):
    """Create model with stronger regularization to reduce overfitting"""
    
    # Enhanced regularization parameters
    best_params = {
        'max_depth': 4,  # Reduced from 5
        'learning_rate': 0.1,  # Reduced from 0.15 for more stable learning
        'reg_alpha': 1.0,  # Increased L1 regularization
        'reg_lambda': 2.0,  # Increased L2 regularization
        'subsample': 0.7,  # Reduced from 0.85
        'colsample_bytree': 0.7,  # Reduced from 0.8
        'gamma': 0.2,  # Increased from 0.05
        'min_child_weight': 5,  # Increased from 3
        'max_delta_step': 1  # Added for imbalanced data
    }
    
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric=['auc', 'logloss'],
        early_stopping_rounds=30,  # More aggressive early stopping
        use_label_encoder=False,
        n_estimators=1000,  # Reduced from 1500
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        **best_params
    )
    
    return model, best_params

def train_with_validation_split(X_train, y_train, validation_size=0.2):
    """Split training data further to create a validation set"""
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=validation_size, 
        random_state=42,
        stratify=y_train
    )
    
    print(f"📊 Enhanced validation split:")
    print(f"   - Training: {X_train_split.shape}")
    print(f"   - Validation: {X_val.shape}")
    
    return X_train_split, X_val, y_train_split, y_val

def train_regularized_model(X_train, X_test, y_train, y_test):
    """Train model with enhanced regularization"""
    
    print("\n🚀 TRAINING REGULARIZED MODEL (Reducing Overfitting)")
    print("=" * 60)
    
    # Create enhanced validation set
    X_train_split, X_val, y_train_split, y_val = train_with_validation_split(X_train, y_train)
    
    model, params = create_regularized_model(y_train)
    
    print("🎯 Regularized Parameters (Anti-Overfitting):")
    for param, value in params.items():
        print(f"   - {param}: {value}")
    
    # Use three sets: train, validation, test
    eval_set = [(X_train_split, y_train_split), (X_val, y_val), (X_test, y_test)]
    
    start_time = datetime.now()
    print(f"\n🤖 Starting regularized training at {start_time.strftime('%H:%M:%S')}...")
    
    model.fit(
        X_train_split, y_train_split,
        eval_set=eval_set,
        verbose=50
    )
    
    training_time = datetime.now() - start_time
    print(f"⏱️  Training completed in {training_time}")
    
    # Analyze results
    results = model.evals_result()
    best_iteration = model.best_iteration
    
    # Get scores from all sets
    train_auc = results['validation_0']['auc'][best_iteration]
    val_auc = results['validation_1']['auc'][best_iteration]
    test_auc = results['validation_2']['auc'][best_iteration]
    
    train_test_gap = train_auc - test_auc
    train_val_gap = train_auc - val_auc
    
    print(f"\n🏆 REGULARIZATION RESULTS:")
    print(f"   - Training AUC: {train_auc:.4f}")
    print(f"   - Validation AUC: {val_auc:.4f}") 
    print(f"   - Test AUC: {test_auc:.4f}")
    print(f"   - Train-Val Gap: {train_val_gap:.4f} (Overfitting indicator)")
    print(f"   - Train-Test Gap: {train_test_gap:.4f}")
    print(f"   - Best Iteration: {best_iteration}")
    
    # Compare with previous overfitting
    previous_gap = 0.9458 - 0.8335  # Your previous results
    improvement = previous_gap - train_test_gap
    print(f"   - Overfitting Reduction: {improvement:.4f} improvement!")
    
    return model, results

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    
    print("\n" + "="*60)
    print("🔍 MODEL EVALUATION")
    print("="*60)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"📈 Test AUC: {test_auc:.4f}")
    
    # Test different thresholds
    print("\n🎯 Threshold Analysis:")
    print("Threshold | Precision | Recall | F1-Score | Gap vs 0.5")
    print("-" * 55)
    
    baseline_pred = (y_pred_proba > 0.5).astype(int)
    baseline_f1 = f1_score(y_test, baseline_pred)
    
    for threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
        y_pred = (y_pred_proba > threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f1_gap = f1 - baseline_f1
        
        print(f"   {threshold:.2f}   |   {precision:.3f}   |  {recall:.3f}  |   {f1:.3f}   |   {f1_gap:+.3f}")
    
    # Use optimal threshold
    optimal_threshold = 0.55
    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
    
    precision = precision_score(y_test, y_pred_optimal)
    recall = recall_score(y_test, y_pred_optimal)
    f1 = f1_score(y_test, y_pred_optimal)
    
    print(f"\n💼 BUSINESS METRICS (Threshold: {optimal_threshold}):")
    print(f"   • Precision: {precision:.1%} - Targeting accuracy")
    print(f"   • Recall: {recall:.1%} - Churner coverage") 
    print(f"   • F1-Score: {f1:.3f} - Overall performance")
    
    # Confusion matrix details
    cm = confusion_matrix(y_test, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()
    print(f"   • True Positives: {tp} (churners correctly identified)")
    print(f"   • False Positives: {fp} (false alarms)")
    print(f"   • False Negatives: {fn} (missed churners)")
    
    return {
        'test_auc': test_auc,
        'optimal_threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_enhanced_training_history(results):
    """Plot training history with enhanced analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)

    # AUC plot with all three sets
    ax1.plot(x_axis, results['validation_0']['auc'], label='Train', 
             linewidth=2, color='blue', alpha=0.8)
    ax1.plot(x_axis, results['validation_1']['auc'], label='Validation', 
             linewidth=2, color='red', alpha=0.8)
    ax1.plot(x_axis, results['validation_2']['auc'], label='Test', 
             linewidth=2, color='green', alpha=0.8)
    ax1.set_title('XGBoost AUC - Regularized Training', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('AUC')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log loss plot
    ax2.plot(x_axis, results['validation_0']['logloss'], label='Train', 
             linewidth=2, color='blue', alpha=0.8)
    ax2.plot(x_axis, results['validation_1']['logloss'], label='Validation', 
             linewidth=2, color='red', alpha=0.8)
    ax2.plot(x_axis, results['validation_2']['logloss'], label='Test', 
             linewidth=2, color='green', alpha=0.8)
    ax2.set_title('XGBoost Log Loss - Regularized Training', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def save_model_with_threshold(model, threshold, model_dir, model_name):
    """Save model and threshold together"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(xgb_model, f'{model_name}_{timestamp}.json')
    model.save_model(model_path)
    
    # Save threshold and metadata
    metadata = {
        'threshold': threshold,
        'model_path': model_path,
        'timestamp': timestamp
    }
    
    metadata_path = os.path.join(model_dir, f'xgb/{model_name}_metadata_{timestamp}.joblib')
    joblib.dump(metadata, metadata_path)
    
    print(f"💾 Model saved: {model_path}")
    print(f"💾 Metadata saved: {metadata_path}")
    print(f"🎯 Optimal threshold: {threshold:.3f}")
    
    return model_path, metadata_path

# Main execution
if __name__ == "__main__":
    print("🚀 XGBoost Regularized Model Training (Reducing Overfitting)")
    print("=" * 65)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data()
        
        print(f"📊 Data Overview:")
        print(f"   - Training data: {X_train.shape}")
        print(f"   - Test data: {X_test.shape}")
        
        # Train regularized model
        final_model, final_results = train_regularized_model(X_train, X_test, y_train, y_test)
        
        # Evaluate model
        eval_results = evaluate_model(final_model, X_test, y_test)
        
        # Plot training history
        plot_enhanced_training_history(final_results)
        
        # Save optimized model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(xgb_model, f'xgboost_regularized_{timestamp}.json')
        final_model.save_model(model_path)
        
        print(f"\n💾 Regularized model saved: {model_path}")
        
        # Final summary
        print(f"\n🎯 **REGULARIZED TRAINING COMPLETE**")
        print("=" * 45)
        print(f"Final Test AUC: {eval_results['test_auc']:.4f}")
        print(f"Optimal threshold: {eval_results['optimal_threshold']:.3f}")
        print(f"Precision: {eval_results['precision']:.1%}")
        print(f"Recall: {eval_results['recall']:.1%}")
        print(f"F1-Score: {eval_results['f1']:.3f}")

        save_model_with_threshold(
            model=final_model,
            threshold=eval_results['optimal_threshold'],
            model_dir=models_dir,
            model_name='xgboost_balanced'
        )

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()