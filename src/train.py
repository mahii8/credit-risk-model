import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              classification_report)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(filepath):
    """Load labeled customer features"""
    df = pd.read_csv(filepath)
    df = df.dropna(subset=['is_high_risk'])
    return df


def get_feature_columns():
    """Return list of features for modeling"""
    return [
        'Recency', 'Frequency', 'Monetary',
        'Avg_Amount', 'Std_Amount', 'Max_Amount', 'Min_Amount',
        'Total_Value', 'Avg_Value', 'Total_Fraud', 'Fraud_Rate',
        'Unique_Products', 'Unique_Channels', 'Unique_Categories',
        'Avg_Hour', 'Avg_DayOfWeek', 'Weekend_Ratio'
    ]


def evaluate_model(model, X_test, y_test, model_name):
    """Compute and return all evaluation metrics"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
        'f1':        round(f1_score(y_test, y_pred), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n{'='*40}")
    print(f"  {model_name} Results")
    print(f"{'='*40}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print(classification_report(y_test, y_pred,
          target_names=['Low Risk','High Risk']))

    return metrics


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with scaling"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(X_train, y_train):
    """Train Random Forest with basic tuning"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier"""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    return model


def run_training():
    os.chdir(r'C:\Users\bamla\OneDrive\Desktop\credit-risk-model')

    # Load data
    print("Loading data...")
    df = load_processed_data('data/processed/customer_features_labeled.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"High-risk rate: {df['is_high_risk'].mean()*100:.1f}%")

    # Features & target
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df['is_high_risk']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # MLflow experiment
    mlflow.set_experiment("credit_risk_model")

    results = {}

    # ── Logistic Regression ─────────────────────────────
    with mlflow.start_run(run_name="LogisticRegression"):
        print("\nTraining Logistic Regression...")
        lr_model = train_logistic_regression(X_train, y_train)
        metrics = evaluate_model(lr_model, X_test, y_test,
                                 "Logistic Regression")
        mlflow.log_params({
            'model': 'LogisticRegression',
            'max_iter': 1000,
            'class_weight': 'balanced'
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr_model, "logistic_regression_model")
        results['Logistic Regression'] = metrics

    # ── Random Forest ────────────────────────────────────
    with mlflow.start_run(run_name="RandomForest"):
        print("\nTraining Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        metrics = evaluate_model(rf_model, X_test, y_test,
                                 "Random Forest")
        mlflow.log_params({
            'model': 'RandomForest',
            'n_estimators': 100,
            'max_depth': 10,
            'class_weight': 'balanced'
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        results['Random Forest'] = metrics

    # ── XGBoost ──────────────────────────────────────────
    with mlflow.start_run(run_name="XGBoost"):
        print("\nTraining XGBoost...")
        xgb_model = train_xgboost(X_train, y_train)
        metrics = evaluate_model(xgb_model, X_test, y_test,
                                 "XGBoost")
        mlflow.log_params({
            'model': 'XGBoost',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        results['XGBoost'] = metrics

    # ── Model Comparison Table ───────────────────────────
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison = pd.DataFrame(results).T
    print(comparison.to_string())

    # Save best model
    best_model_name = comparison['roc_auc'].idxmax()
    best_roc = comparison['roc_auc'].max()
    print(f"\n✅ Best model: {best_model_name} (ROC-AUC: {best_roc:.4f})")

    # Save comparison table
    comparison.to_csv('data/processed/model_comparison.csv')
    print("Comparison saved to data/processed/model_comparison.csv")
    # SHAP Analysis on best model (Random Forest)
    print("\n[SHAP] Computing feature importance...")
    import shap
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    shap_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_SHAP': np.abs(shap_values[:,:,1]).mean(axis=0)
    }).sort_values('Mean_SHAP', ascending=False)

    print("\nTop 10 Features by SHAP:")
    print(shap_importance.head(10).to_string(index=False))

    # Save SHAP plot
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values[:,:,1], X_test,
                      feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig('notebooks/plot_shap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved!")

    return results, best_model_name


if __name__ == '__main__':
    results, best = run_training()
