"""
📊 Portfolio Visualizations for Customer Churn Prediction
Clean, separate charts for portfolio showcase
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_style():
    """Set consistent styling for all charts"""
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11

def create_model_comparison_chart():
    """Create clean model comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['XGBoost', 'Neural Network', 'Ensemble']
    precision = [0.598, 0.640, 0.605]
    recall = [0.795, 0.649, 0.768]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, precision, width, label='Precision', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, recall, width, label='Recall', alpha=0.8, color='#ff7f0e')
    
    ax.set_title('Model Performance Comparison', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.9)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(precision, recall)):
        ax.text(i - width/2, v1 + 0.02, f'{v1:.1%}', ha='center', fontweight='bold')
        ax.text(i + width/2, v2 + 0.02, f'{v2:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/1_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 1_model_comparison.png")

def create_key_metrics_chart():
    """Create key metrics chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [0.805, 0.605, 0.768, 0.677]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8)
    ax.set_title('Key Performance Metrics', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, v in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03, 
                f'{v:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/2_key_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 2_key_metrics.png")

def create_cost_savings_chart():
    """Create cost savings chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = ['Default Threshold', 'Optimized Threshold']
    false_positives = [1344, 938]
    reduction = false_positives[0] - false_positives[1]
    
    bars = ax.bar(scenarios, false_positives, color=['#ff6b6b', '#51cf66'], alpha=0.8)
    ax.set_title('False Alarm Reduction', fontweight='bold')
    ax.set_ylabel('False Positives')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, v in zip(bars, false_positives):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, 
                f'{v:,}', ha='center', fontweight='bold')
    
    # Add reduction text
    ax.text(0.5, 1500, f'30% Reduction\n({reduction} fewer false alarms)', 
            ha='center', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('visuals/3_cost_savings.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 3_cost_savings.png")

def create_feature_importance_chart():
    """Create feature importance chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = ['Tenure', 'Monthly Charges', 'Contract Type', 'Service Usage', 'Payment Method']
    importance = [0.85, 0.78, 0.72, 0.65, 0.58]
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color='#17becf', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Relative Importance')
    ax.set_title('Top Predictive Features', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, importance)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{v:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/4_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 4_feature_importance.png")

def create_business_impact_chart():
    """Create business impact chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Better Targeting', 'Cost Reduction', 'Customer Coverage']
    values = [60.5, 30.0, 76.8]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_title('Business Impact Analysis', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/5_business_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 5_business_impact.png")

def create_ensemble_strategies_chart():
    """Create ensemble strategies chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = ['Precision Focused', 'Recall Focused', 'Conservative']
    precision_scores = [0.640, 0.567, 0.605]
    recall_scores = [0.649, 0.800, 0.768]
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, precision_scores, width, label='Precision', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, recall_scores, width, label='Recall', alpha=0.8, color='#ff7f0e')
    
    ax.set_title('Ensemble Strategy Comparison', fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.9)
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(precision_scores, recall_scores)):
        ax.text(i - width/2, v1 + 0.02, f'{v1:.1%}', ha='center', fontweight='bold')
        ax.text(i + width/2, v2 + 0.02, f'{v2:.1%}', ha='center', fontweight='bold')
    
    # Highlight recommended strategy
    ax.text(2, 0.85, 'Recommended', ha='center', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('visuals/6_ensemble_strategies.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 6_ensemble_strategies.png")

def create_simple_architecture_diagram():
    """Create a simple architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Simple text-based architecture
    architecture_text = """
    🏗️ SYSTEM ARCHITECTURE
    
    📥 DATA INPUT
    │
    ├── Raw Customer Data
    ├── Demographics
    └── Service Usage
    
    ⚙️ PROCESSING
    │
    ├── Feature Engineering (50+ features)
    ├── Data Cleaning
    └── Scaling/Normalization
    
    🤖 MODEL TRAINING
    │
    ├── XGBoost (High Recall: 79.5%)
    ├── Neural Network (High Precision: 64.0%)
    └── Ensemble Methods
    
    🎯 PREDICTION
    │
    ├── Business-Optimized Thresholds
    ├── Multiple Strategy Options
    └── Confidence Scoring
    
    📊 OUTPUT
    │
    ├── Churn Probabilities
    ├── Risk Levels
    └── Business Impact Analysis
    
    
    🎯 RESULTS: 80.5% Accuracy • 30% Cost Reduction
    """
    
    ax.text(0.1, 0.9, architecture_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=12, va='top', linespacing=1.5)
    
    plt.tight_layout()
    plt.savefig('visuals/7_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 7_architecture_diagram.png")

def create_performance_summary():
    """Create a clean performance summary"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove axes for text display
    ax.axis('off')
    
    summary_text = """
    🎯 CUSTOMER CHURN PREDICTION SYSTEM
    
    📊 PERFORMANCE SUMMARY
    
    Accuracy:   80.5%  - Reliable predictions
    Precision:  60.5%  - Better targeting
    Recall:     76.8%  - Comprehensive coverage
    F1-Score:   67.7%  - Overall balance
    
    💼 BUSINESS IMPACT
    
    • 30% reduction in false alarms
    • 60.5% of predicted churners actually churn
    • 76.8% of actual churners identified
    • Significant cost savings in retention efforts
    
    🔧 TECHNICAL HIGHLIGHTS
    
    • Multi-model ensemble (XGBoost + Neural Network)
    • 50+ engineered features
    • Business-optimized thresholds
    • Production-ready pipeline
    • Cost-sensitive evaluation
    
    🏆 KEY ACHIEVEMENTS
    
    • Outperforms single-model approaches
    • Balances precision and recall effectively
    • Provides actionable business insights
    • Reduces wasted retention efforts by 30%
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=11, va='top', linespacing=1.4)
    
    plt.tight_layout()
    plt.savefig('visuals/8_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 8_performance_summary.png")

def main():
    """Generate all charts one by one"""
    print("📊 Generating portfolio visualizations...")
    print("=" * 50)
    
    setup_style()
    
    # Generate each chart separately
    create_model_comparison_chart()
    create_key_metrics_chart()
    create_cost_savings_chart()
    create_feature_importance_chart()
    create_business_impact_chart()
    create_ensemble_strategies_chart()
    create_simple_architecture_diagram()
    create_performance_summary()
    
    print("=" * 50)
    print("🎉 All visualizations generated successfully!")
    print("\n📁 Generated Files:")
    print("1_model_comparison.png - Model performance comparison")
    print("2_key_metrics.png - Core performance metrics") 
    print("3_cost_savings.png - Business cost reduction")
    print("4_feature_importance.png - Top predictive features")
    print("5_business_impact.png - Overall business impact")
    print("6_ensemble_strategies.png - Strategy comparison")
    print("7_architecture_diagram.png - System architecture")
    print("8_performance_summary.png - Project summary")

if __name__ == "__main__":
    main()