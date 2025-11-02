# Customer Churn Prediction System 

## 🎯 Project Description

Built an advanced machine learning system that predicts customer churn with 80.5% accuracy. The system combines XGBoost and Neural Networks with smart ensemble methods to deliver business-optimized predictions, reducing false alarms by 30% while maintaining comprehensive churner coverage.

## 🚀 Key Achievements

- **80.5% Accuracy** on imbalanced customer data
- **60.5% Precision** - Better targeting of at-risk customers
- **76.8% Recall** - Comprehensive churner identification  
- **30% Reduction** in false alarms through optimized thresholds
- **Production-ready** pipeline with modular design

## 🛠️ Technical Implementation

### Architecture
- **Multi-model approach**: XGBoost (high recall) + Neural Networks (high precision)
- **Smart ensemble strategies**: Precision-focused, recall-focused, and balanced approaches
- **Business-optimized thresholds**: Cost-sensitive learning beyond default 0.5 threshold
- **Advanced feature engineering**: 50+ features capturing customer behavior and value

### Technical Stack
- **Machine Learning**: XGBoost, TensorFlow, Scikit-learn
- **Data Processing**: Pandas, NumPy, Feature Engineering
- **Model Optimization**: Hyperparameter tuning, Cross-validation, Ensemble methods
- **Production Features**: Model persistence, Error handling, Modular pipeline

## 💼 Business Impact

![Model Comparison](https://github.com/Haseeb2510/Churn-Prediction/blob/main/visuals/5_business_impact.png)

### Cost Savings
- **30% reduction in false positives** = Fewer wasted retention efforts
- **60.5% precision** = Better targeting of retention campaigns
- **76.8% recall** = Comprehensive coverage of potential churners

### Use Cases
- **Proactive retention**: Identify at-risk customers before they leave
- **Resource optimization**: Focus retention budget on high-probability churners
- **Customer insights**: Understand drivers of churn behavior

## 📈 Performance Metrics

| Metric | XGBoost | Neural Network | Ensemble |
|--------|---------|----------------|----------|
| Accuracy | 80.4% | 81.0% | 80.5% |
| Precision | 59.8% | 64.0% | 60.5% |
| Recall | 79.5% | 64.9% | 76.8% |
| F1-Score | 0.682 | 0.644 | 0.677 |

![Model Comparison](https://github.com/Haseeb2510/Churn-Prediction/blob/main/visuals/1_model_comparison.png)

## 🔧 Key Features

1. **Business-Optimized Thresholds**
   - Moved beyond technical metrics to business impact
   - Cost-sensitive optimization considering false positive costs
   - Multiple threshold strategies for different business objectives

2. **Advanced Ensemble Methods**
   - Precision-focused: Maximizes prediction accuracy (64.0% precision)
   - Recall-focused: Maximizes churner coverage (80.0% recall)  
   - Conservative: Balanced approach (recommended - 60.5% precision, 76.8% recall)

3. **Production-Ready Pipeline**
   - Modular design for maintainability
   - Comprehensive error handling and validation
   - Model versioning and persistence
   - Real-time prediction capabilities

## 🎯 Technical Challenges Overcome

1. **Class Imbalance**: Implemented scale_pos_weight and class weights
2. **Overfitting**: Used regularization, dropout, and early stopping
3. **Business Alignment**: Developed cost-sensitive evaluation metrics
4. **Model Diversity**: Combined tree-based and neural network approaches

## 📁 Project Structure

```
customer-churn-prediction/
│
├── 📁 analysis/                 # Model training visualizations
│
├── 📁 data/                     # Data directory (add your data here)
│   ├── raw/                     # Raw customer data
│   ├── worked/                  # Processed data
│   └── new/                     # New customers for prediction
│
├── 📁 logs/                     # Tensorboard visualization
│   ├── train/                   # Train visualization
│   └── validation/              # Validation visualization
│
├── 📁 model/                    # Trained models & metadata
│   ├── xgb/                     # XGBoost models
│   └── nn/                      # Neural Network models
│
├── 📁 notebook/                
│   └── predict_new.py           # Explains the end-to-end pipeline
│
├── 📁 prediction/
│   └── predict_new.py           # Production prediction system
│
├── 📁 src/
│   ├── xgb_training.py          # XGBoost model training & optimization
│   ├── nn_training.py           # Neural Network architecture & training
│   └── utils.py                 # Data preprocessing & feature engineering
│   └── data_analysis.py         # Analyzing Data
│
├── 📁 visuals/                  # Project visualization
│
├── demo.py                      # Quick demo script
├── visuals.py                   # Quick visualization of models
├── requirements.txt             # Python dependencies
└── setup_guide.md            
```
## 🏆 Skills Demonstrated

- **Machine Learning**: Ensemble methods, Neural Networks, XGBoost, Hyperparameter tuning
- **Software Engineering**: Modular design, Production pipelines, Error handling
- **Business Acumen**: Cost-sensitive learning, ROI optimization, Business metrics
- **Data Science**: Feature engineering, Model evaluation, Imbalanced data handling
## 📊 Business Impact

### 💰 Cost Savings
- **Reduced false positives**: 30% fewer wasted retention efforts
- **Better targeting**: 60.5% of predicted churners actually churn
- **Comprehensive coverage**: 76.8% of actual churners identified

### 🎯 Use Cases
- **Proactive retention**: Identify at-risk customers before they leave
- **Resource optimization**: Focus retention efforts on high-probability churners  
- **Customer segmentation**: Understand drivers of churn behavior
- **Product improvement**: Identify service gaps causing churn

## 🔬 Model Performance

### Individual Models
| Metric | XGBoost | Neural Network |
|--------|---------|----------------|
| **Accuracy** | 80.4% | 81.0% |
| **Precision** | 59.8% | 64.0% |
| **Recall** | 79.5% | 64.9% |
| **F1-Score** | 0.682 | 0.644 |

### Ensemble Performance (Conservative)
| Metric | Value | Business Impact |
|--------|-------|-----------------|
| **Accuracy** | 80.5% | Reliable predictions |
| **Precision** | 60.5% | Better targeting |
| **Recall** | 76.8% | Good coverage |
| **False Alarms** | 938 | 30% reduction vs baseline |

![Model Comparison](https://github.com/Haseeb2510/Churn-Prediction/blob/main/visuals/6_ensemble_strategies.png)

### You can view the full training notebook here:

[📓 Model Training Notebook](https://github.com/Haseeb2510/Churn-Prediction/blob/main/notebooks/01_Customer_Churn_Training.ipynb)

## 🚦 Getting Started

### 1. Prerequisites
```bash
Python 3.8+
Required packages in requirements.txt
```

### 2. Data Preparation
Place your customer data in `data/raw/` following the expected format.

### 3. Training
Run the training scripts to build models optimized for your data.

### 4. Prediction
Use the prediction system to identify at-risk customers.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Abdul Haseeb**
- GitHub: [@Haseeb2510](https://github.com/Haseeb2510)
- LinkedIn: [Abdul Haseeb](https://www.linkedin.com/in/haseeb-abdul-172542243)

## 🎉 Acknowledgments

- Telco Customer Churn dataset
- XGBoost and TensorFlow communities
- Scikit-learn for robust ML utilities

