"""
🚀 Customer Churn Prediction - Quick Ensemble Demo
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prediction.predict_new import (
    load_xgb_model,
    load_nn_model,
    create_ensemble
)

# ────────────────────────────────────────────────
# 🔹 Quick demo showing how the ensemble model works
# ────────────────────────────────────────────────

def quick_demo():
    print("\n🚀 CUSTOMER CHURN ENSEMBLE DEMO")
    print("=" * 60)
    print("Combining XGBoost + Neural Network for improved churn prediction.")
    print("=" * 60)

    try:
        # 1️⃣ Load trained models
        print("\n📦 Loading models...")
        xgb_model = load_xgb_model()
        nn_model = load_nn_model()

        # 2️⃣ Create ensemble
        method = "conservative"
        ensemble = create_ensemble(xgb_model, nn_model, method)
        print(f"✅ Ensemble method: {method.upper()}")

        # 3️⃣ Load and sample data
        data_path = os.path.join(os.path.dirname(__file__), "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df = pd.read_csv(data_path)
        sample_df = df.sample(10, random_state=42).reset_index(drop=True)
        print(f"\n🧾 Using {len(sample_df)} random customers for demo")

        # 4️⃣ Preprocess data using one of the models (for consistency)
        processed_df = xgb_model.preprocess_new_data(sample_df)

        # 5️⃣ Get predictions from each model
        xgb_preds = xgb_model.predict_data(processed_df)
        nn_preds = nn_model.predict_data(processed_df)
        ens_preds = ensemble.predict_proba(processed_df)

        # 6️⃣ Display sample comparison
        print("\n📊 MODEL PREDICTION COMPARISON")
        print("=" * 60)
        for i in range(len(sample_df)):
            cid = sample_df.iloc[i].get("customerID", f"Client_{i+1}")
            print(f"🧍‍♂️ {cid}:")
            print(f"   XGBoost   → {xgb_preds[i]:.3f}")
            print(f"   NeuralNet → {nn_preds[i]:.3f}")
            print(f"   Ensemble  → {ens_preds[i]:.3f}")
            print(f"   Final Risk → {'🚨 CHURN RISK' if ens_preds[i] > 0.55 else '✅ LOW RISK'}")
            print("-" * 60)

        # 7️⃣ Visualization
        visualize_results(sample_df, xgb_preds, nn_preds, ens_preds)

        print("\n✅ Demo completed successfully!")
        print("💡 Ensemble combines both models' strengths for better balance.")
        print("   - High recall from XGBoost")
        print("   - High precision from Neural Network")
        print("   - Fewer false alarms with Conservative method\n")

    except Exception as e:
        print(f"❌ Demo Error: {e}")
        print("💡 Make sure trained models and encoders are available under /model/")
        return


# ────────────────────────────────────────────────
# 📊 Visualization helper
# ────────────────────────────────────────────────
def visualize_results(df, xgb_preds, nn_preds, ens_preds):
    """Visualize predictions from XGBoost, NeuralNet, and Ensemble"""
    try:
        customers = [df.iloc[i].get("customerID", f"C{i+1}") for i in range(len(df))]
        x = np.arange(len(customers))
        width = 0.25

        plt.figure(figsize=(10, 5))
        plt.bar(x - width, xgb_preds, width, label="XGBoost", alpha=0.7)
        plt.bar(x, nn_preds, width, label="NeuralNet", alpha=0.7)
        plt.bar(x + width, ens_preds, width, label="Ensemble", alpha=0.7)

        plt.axhline(0.55, color="red", linestyle="--", label="Threshold (0.55)")
        plt.xticks(x, customers, rotation=45, ha="right")
        plt.ylabel("Predicted Churn Probability")
        plt.title("Model Prediction Comparison per Customer")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Visualization skipped due to: {e}")


# ────────────────────────────────────────────────
# Run demo
# ────────────────────────────────────────────────
if __name__ == "__main__":
    quick_demo()
