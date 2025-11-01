import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.metrics import AUC, Precision, Recall, TruePositives, FalsePositives # type: ignore
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os,joblib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

models_dir = os.path.join(project_root, "model")
os.makedirs(models_dir, exist_ok=True)

nn_model = os.path.join(project_root, "model/nn")
os.makedirs(nn_model, exist_ok=True)


def load_data():
     
    X_train = joblib.load(os.path.join(models_dir, "X_train.joblib"))
    X_test = joblib.load(os.path.join(models_dir, "X_test.joblib"))
    y_train = joblib.load(os.path.join(models_dir, "y_train.joblib")) 
    y_test = joblib.load(os.path.join(models_dir, "y_test.joblib"))

    return X_train, X_test, y_train, y_test

def create_nn(input_dim):
    # Input layer with advanced regularization
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.002), kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.5),

        # Hidden, layers with decreasing complexity
        Dense(256, activation='elu', kernel_regularizer=l2(0.0015), kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(128, activation='elu', kernel_regularizer=l2(0.001), kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='elu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='elu'),
        BatchNormalization(),
        Dropout(0.1),

        Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')
    ])

    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        amsgrad=True 
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            TruePositives(name='tp'),
            FalsePositives(name='fp')
        ]
    )
    
    print(f"🧠 Neural Network: {input_dim} input features")
    print(f"   - Architecture: 512-256-128-64-32-1")
    print(f"   - Total parameters: {model.count_params():,}")

    return model

def calculate_enhanced_class_weights(y_train):
    """Robust class weight calculation with negative value handling"""
    
    # Ensure 1D array and convert to integer
    y_flat = y_train.reshape(-1) if len(y_train.shape) > 1 else y_train
    y_int = y_flat.astype(int)
    
    # DEBUG: Check for negative values
    print(f"🔍 y_int min: {np.min(y_int)}, max: {np.max(y_int)}")
    print(f"🔍 y_int unique values: {np.unique(y_int)}")
    
    # Check for negative values and fix if needed
    if np.min(y_int) < 0:
        print(f"⚠️ Negative values detected! Shifting to non-negative...")
        y_int = y_int - np.min(y_int)  # Shift to make all values >= 0
        print(f"✅ Fixed y_int unique values: {np.unique(y_int)}")
    
    # Get unique classes
    unique_classes = np.unique(y_int)
    
    # Get class distribution
    class_counts = np.bincount(y_int)
    total_samples = len(y_int)
    n_classes = len(unique_classes)
    
    print(f"🔍 Class distribution: {class_counts}")
    print(f"🔍 Total samples: {total_samples}")
    
    # Calculate balanced class weights
    weights = total_samples / (n_classes * class_counts)
    
    # Normalize to make minority class weight ~2.0
    normalized_weights = weights / np.min(weights)
    
    # Create dictionary - map back to original classes if we shifted
    if np.min(y_train.astype(int)) < 0:
        # If we shifted from [-1, 0] to [0, 1], map back
        original_min = np.min(y_train.astype(int))
        class_weights = {
            int(cls + original_min): float(normalized_weights[cls]) 
            for cls in unique_classes
        }
    else:
        class_weights = {
            int(cls): float(normalized_weights[cls]) 
            for cls in unique_classes
        }
    
    print(f"🎯 Calculated class weights: {class_weights}")
    return class_weights

def create_advanced_callbacks(checkpoint_dir):
    """Create comprehensive callbacks for better training control"""
    
    # Model checkpoint - using .keras format
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( #type:ignore
        filepath=os.path.join(checkpoint_dir, 'best_model_epoch_{epoch:02d}_auc_{val_auc:.4f}.keras'),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Enhanced early stopping
    early_stopping = EarlyStopping(
        monitor='val_auc',
        patience=40,
        restore_best_weights=True,
        mode='max',
        min_delta=0.0005,
        verbose=1,
        baseline=None
    )
    
    # Dynamic learning rate reduction - works with static learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=12,
        min_lr=1e-8,
        mode='max',
        min_delta=0.001,
        cooldown=2,
        verbose=1
    )
    
    # TensorBoard for visualization (optional)
    tensorboard_callback = tf.keras.callbacks.TensorBoard( #type:ignore
        log_dir=os.path.join(project_root, 'logs'),
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq='epoch'
    )
    
    return [checkpoint_callback, early_stopping, reduce_lr, tensorboard_callback]

def train_neural_network(X_train, X_test, y_train, y_test):
    input_dim = X_train.shape[1]
    model = create_nn(input_dim)
    
    print("🚀 Optimized Neural Network Architecture:")
    model.summary()
    
    # Calculate enhanced class weights
    class_weights = calculate_enhanced_class_weights(y_train)
    print(f"🎯 Enhanced class weights: {class_weights}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(models_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Advanced callbacks
    callbacks = create_advanced_callbacks(checkpoint_dir)
    
    # Dynamic batch size based on dataset size
    dataset_size = X_train.shape[0]
    if dataset_size > 10000:
        batch_size = 128
    elif dataset_size > 5000:
        batch_size = 64
    else:
        batch_size = 32
    
    print(f"🤖 Starting training with batch_size={batch_size}")
    
    # Train with validation split for additional monitoring
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,  # Slightly more epochs for complex architecture
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
        shuffle=True,  # Ensure data is shuffled
        validation_freq=1  # Validate every epoch
    )
    
    # Print training summary
    best_epoch = np.argmax(history.history['val_auc'])
    print(f"\n🏆 Training Summary:")
    print(f"   - Best epoch: {best_epoch + 1}")
    print(f"   - Best validation AUC: {history.history['val_auc'][best_epoch]:.4f}")
    print(f"   - Best validation loss: {history.history['val_loss'][best_epoch]:.4f}")
    
    return model, history

def find_realistic_thresholds(y_test, y_pred_proba):
    """Find optimal thresholds with realistic analysis"""
    thresholds = np.linspace(0.1, 0.9, 17)  # From 0.1 to 0.9 in 0.05 steps
    
    print("🎯 Realistic Threshold Analysis:")
    print("Threshold | Precision | Recall | F1-Score | TP | FP")
    print("-" * 60)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"   {threshold:.2f}   |   {precision:.3f}   |  {recall:.3f}  |  {f1:.3f}   | {tp:3d} | {fp:3d}")
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp
        })
    
    # Find optimal threshold by different criteria
    results_df = pd.DataFrame(results)
    
    # Best F1 score
    best_f1_idx = results_df['f1'].idxmax()
    best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
    
    # Best balanced (precision ~65%, recall reasonable)
    balanced_mask = (results_df['precision'] >= 0.60) & (results_df['recall'] >= 0.60)
    if balanced_mask.any():
        balanced_df = results_df[balanced_mask]
        best_balanced_idx = balanced_df['f1'].idxmax()
        best_balanced_threshold = balanced_df.loc[best_balanced_idx, 'threshold']
    else:
        best_balanced_threshold = 0.5
    
    print(f"\n🏆 Best F1 Threshold: {best_f1_threshold:.3f} (F1: {results_df.loc[best_f1_idx, 'f1']:.3f})")
    print(f"⚖️  Balanced Threshold: {best_balanced_threshold:.3f} (Precision: {results_df.loc[results_df['threshold'] == best_balanced_threshold, 'precision'].values[0]:.3f}, Recall: {results_df.loc[results_df['threshold'] == best_balanced_threshold, 'recall'].values[0]:.3f})")
    
    return results_df, best_f1_threshold, best_balanced_threshold

def get_business_recommendation(results_df):
    """Get practical business recommendation for threshold"""
    
    # Business-oriented criteria - prefer higher precision while maintaining good recall
    business_mask = (results_df['precision'] >= 0.53) & (results_df['recall'] >= 0.75)
    
    if business_mask.any():
        business_df = results_df[business_mask]
        # Prefer higher precision within this range
        best_business_idx = business_df['precision'].idxmax()
        best_business_threshold = business_df.loc[best_business_idx, 'threshold']
    else:
        # Fall back to best F1
        best_business_threshold = results_df.loc[results_df['f1'].idxmax(), 'threshold']
    
    # Get metrics for recommended threshold
    recommended_metrics = results_df[results_df['threshold'] == best_business_threshold].iloc[0]
    
    print("\n💼 **BUSINESS RECOMMENDATION** 💼")
    print("=" * 50)
    print(f"🎯 Recommended Threshold: {best_business_threshold:.3f}")
    print(f"📈 Expected Performance:")
    print(f"   • Precision: {recommended_metrics['precision']:.1%} (churn predictions that are correct)")
    print(f"   • Recall: {recommended_metrics['recall']:.1%} (actual churners caught)")
    print(f"   • F1-Score: {recommended_metrics['f1']:.3f} (overall balance)")
    print(f"   • True Positives: {recommended_metrics['tp']}/374 (churners identified)")
    print(f"   • False Positives: {recommended_metrics['fp']} (false alarms)")
    
    # Compare with default threshold
    default_metrics = results_df[results_df['threshold'] == 0.5].iloc[0]
    fp_reduction = default_metrics['fp'] - recommended_metrics['fp']
    
    print(f"\n📊 Business Impact vs Default 0.5:")
    print(f"   • Precision: +{recommended_metrics['precision'] - default_metrics['precision']:.1%} better targeting")
    print(f"   • Recall: -{default_metrics['recall'] - recommended_metrics['recall']:.1%} fewer churners caught")
    print(f"   • False Alarms: {fp_reduction} fewer wasted retention efforts!")
    
    return best_business_threshold

def plot_threshold_tradeoff(y_test, y_pred_proba):
    """Visualize precision-recall tradeoff"""
    thresholds = np.linspace(0.1, 0.9, 17)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precision-Recall tradeoff
    ax1.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2, marker='o')
    ax1.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2, marker='s')
    ax1.set_xlabel('Classification Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision-Recall Trade-off')
    ax1.legend()
    ax1.grid(True)
    ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Default 0.5')
    
    # F1 score across thresholds
    ax2.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2, marker='^')
    ax2.set_xlabel('Classification Threshold')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score vs Threshold')
    ax2.grid(True)
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Default 0.5')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def simplified_output(results, y_test, model, history):
    """Clean, professional output for LinkedIn/business audience"""
    
    # Get best epoch info
    best_epoch = np.argmax(history.history['val_auc']) + 1
    best_auc = history.history['val_auc'][best_epoch - 1]
    
    print("=" * 60)
    print("🎯 CUSTOMER CHURN PREDICTION - RESULTS SUMMARY")
    print("=" * 60)
    
    # Model Performance
    print("\n📊 MODEL PERFORMANCE")
    print(f"   • Best Validation AUC: {best_auc:.4f}")
    print(f"   • Best Epoch: {best_epoch}")
    print(f"   • Final Test AUC: {results['auc']:.4f}")
    
    # Business Metrics
    business_pred = results['y_pred_business']
    precision = precision_score(y_test, business_pred)
    recall = recall_score(y_test, business_pred)
    f1 = f1_score(y_test, business_pred)
    
    print(f"\n💼 BUSINESS METRICS (Threshold: {results['threshold']:.3f})")
    print(f"   • Precision: {precision:.1%} - Churn predictions that are correct")
    print(f"   • Recall: {recall:.1%} - Actual churners identified") 
    print(f"   • F1-Score: {f1:.3f} - Overall balance")
    
    # Impact Analysis
    default_pred = (results['y_pred_proba'] > 0.5).astype(int)
    tp_business = business_pred.sum()
    tp_default = default_pred.sum()
    fp_reduction = default_pred.sum() - business_pred.sum()
    
    print(f"\n🚀 BUSINESS IMPACT vs Default 0.5 Threshold")
    print(f"   • Better Targeting: +{(precision - precision_score(y_test, default_pred)):+.1%} precision")
    print(f"   • Fewer False Alarms: {fp_reduction} reduced wasted efforts")
    print(f"   • Churners Caught: {tp_business}/{(y_test == 1).sum()} identified")
    
    # Top Features (simplified)
    print(f"\n🔍 TOP 5 PREDICTIVE FACTORS")
    # Get your existing feature importance
    weights = model.layers[0].get_weights()[0]
    importance = np.mean(np.abs(weights), axis=1)
    top_indices = np.argsort(importance)[-5:][::-1]
    
    feature_names = [f'Feature_{i+1}' for i in range(len(importance))]
    for i, idx in enumerate(top_indices, 1):
        print(f"   {i}. {feature_names[idx]}")

def plot_simplified_training(history):
    """Clean training visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax1.set_title('Model Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC
    ax2.plot(history.history['auc'], label='Train', linewidth=2)
    ax2.plot(history.history['val_auc'], label='Validation', linewidth=2)
    ax2.set_title('Model AUC', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def enhanced_evaluation(model, X_test, y_test, history):
    """Comprehensive model evaluation"""
    
    y_pred_proba = model.predict(X_test).flatten()
    y_pred_default = (y_pred_proba > 0.5).astype(int)

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # print(f"🎯 Neural Network AUC: {auc:.4f}")
    
    # Realistic threshold analysis
    results_df, best_f1_threshold, best_balanced_threshold = find_realistic_thresholds(y_test, y_pred_proba)
    business_threshold = get_business_recommendation(results_df)

    # Create predictions with optimal thresholds
    y_pred_best_f1 = (y_pred_proba > best_f1_threshold).astype(int)
    y_pred_balanced = (y_pred_proba > best_balanced_threshold).astype(int)
    y_pred_business = (y_pred_proba > business_threshold).astype(int)

    # print("\n" + "="*60)
    # print("📊 COMPREHENSIVE MODEL EVALUATION")
    # print("="*60)
    
    # print(f"\n📋 Classification Report (Default Threshold 0.5):")
    # print(classification_report(y_test, y_pred_default))
    
    # print(f"\n📋 Classification Report (Best F1 Threshold {best_f1_threshold:.3f}):")
    # print(classification_report(y_test, y_pred_best_f1))
    
    # print(f"\n📋 Classification Report (Balanced Threshold {best_balanced_threshold:.3f}):")
    # print(classification_report(y_test, y_pred_balanced))

    # print(f"\n📋 Classification Report (Business Threshold {business_threshold:.3f}):")
    # print(classification_report(y_test, y_pred_business))

    # Plot threshold analysis
    # plot_threshold_tradeoff(y_test, y_pred_proba)
    
    # Plot training history
    # plot_enhanced_training_history(history)
    
    # Feature importance analysis
    # analyze_feature_importance(model, X_test)
    
    results =  {
        'y_pred_proba': y_pred_proba,
        'y_pred_default': y_pred_default,
        'y_pred_best_f1': y_pred_best_f1,
        'y_pred_balanced': y_pred_balanced,
        'best_f1_threshold': best_f1_threshold,
        'best_balanced_threshold': best_balanced_threshold,
        'auc': auc,
        'y_pred_business': y_pred_business,
        'threshold': business_threshold,
        'results_df': results_df
    }
    
    simplified_output(results, y_test, model, history)
    plot_simplified_training(history)

    return results

def plot_enhanced_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    
    # AUC
    axes[0, 1].plot(history.history['auc'], label='Training AUC')
    axes[0, 1].plot(history.history['val_auc'], label='Validation AUC')
    axes[0, 1].set_title('Model AUC')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(model, feature_names=None, top_k=10):
    """Analyze which features the model considers most important"""
    # Get weights from first layer
    weights = model.layers[0].get_weights()[0]
    
    # Calculate feature importance as average absolute weight
    importance = np.mean(np.abs(weights), axis=1)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importance))]
    
    # Create importance dataframe
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_k)
    
    print(f"\n🔍 Top {top_k} Most Important Features:")
    print(imp_df)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x='importance', y='feature')
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.show()

def main():
    # Main execution
    X_train, X_test, y_train, y_test = load_data()

    print("=== DATA CARDINALITY CHECK ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # They should match:
    # X_train.shape[0] == y_train.shape[0]
    # X_test.shape[0] == y_test.shape[0]

    if X_train.shape[0] != y_train.shape[0]:
        print(f"❌ MISMATCH: X_train has {X_train.shape[0]} samples, but y_train has {y_train.shape[0]} samples")
        
    if X_test.shape[0] != y_test.shape[0]:
        print(f"❌ MISMATCH: X_test has {X_test.shape[0]} samples, but y_test has {y_test.shape[0]} samples")

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Target distribution - Train: {np.unique(y_train, return_counts=True)}")
    print(f"Target distribution - Test: {np.unique(y_test, return_counts=True)}")

    model, history = train_neural_network(X_train, X_test, y_train, y_test)

    results = enhanced_evaluation(model, X_test, y_test, history)

    # Update this part at the very end of your code:
    print(f"\n💡 **FINAL RECOMMENDATION**: Use threshold {results['threshold']:.3f} for optimal business performance")
    print(f"   - Precision: {precision_score(y_test, results['y_pred_business']):.1%} (better targeting)")
    print(f"   - Recall: {recall_score(y_test, results['y_pred_business']):.1%} (good churner coverage)")
    print(f"   - Compared to 0.5: {results['y_pred_default'].sum() - results['y_pred_business'].sum()} fewer false alarms!")

    # Save the recommended threshold for production use
    model_path = os.path.join(project_root, 'model/nn')
    os.makedirs(model_path, exist_ok=True)
    model.save(f'{model_path}/nn_model.h5')
    joblib.dump(results, os.path.join(model_path, 'meta_data.joblib'))
    print(f"💾 Metadata saved to: {os.path.join(model_path, 'nn/optimal_threshold.joblib')}")
    print(f"💾 Model saved to: {os.path.join(model_path, 'nn/model.')}")

if __name__ == '__main__':
    main()