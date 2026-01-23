import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_curve
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    print(f"Evaluating {model_name}...")
    probs = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, probs)
    print(f"PR-AUC: {ap}")

    prec, rec, th = precision_recall_curve(y_test, probs)

    # Recall@Precision >= 90%
    precision_threshold = 0.9
    valid_points = [(r, p) for p, r in zip(prec, rec) if p >= precision_threshold]
    rec_at_prec_90 = max([r for (r, p) in valid_points])
    print(f"Recall at Precision >= {precision_threshold*100}%: {rec_at_prec_90}")

    # Standard metrics at threshold 0.5
    pred = (probs >= 0.5).astype(int)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print(f"F1-Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print("-"*40)

def evaluate_models_to_dataframe(models_dict, X_test, y_test, top_k_percent=0.01):
    """
    Evaluate multiple models and return results in a DataFrame.
    
    Args:
        models_dict (dict): Dictionary with model names as keys and model objects as values
        X_test: Test features
        y_test: Test labels
        top_k_percent (float): Percentage of top predictions to consider for Lift calculation (default 1%)
    
    Returns:
        pd.DataFrame: DataFrame containing all evaluation metrics for each model
    """
    results = []
    
    for model_name, model in models_dict.items():
        # Get predictions
        probs = model.predict_proba(X_test)[:, 1]
        pred = (probs >= 0.5).astype(int)
        
        # Calculate PR-AUC
        ap = average_precision_score(y_test, probs)
        
        # Calculate Precision-Recall curve
        prec, rec, th = precision_recall_curve(y_test, probs)
        
        # Calculate Recall@Precision >= 90%
        precision_threshold_90 = 0.9
        valid_points_90 = [(r, p) for p, r in zip(prec, rec) if p >= precision_threshold_90]
        rec_at_prec_90 = max([r for (r, p) in valid_points_90]) if valid_points_90 else 0.0
        
        # Calculate Recall@Precision >= 95%
        precision_threshold_95 = 0.95
        valid_points_95 = [(r, p) for p, r in zip(prec, rec) if p >= precision_threshold_95]
        rec_at_prec_95 = max([r for (r, p) in valid_points_95]) if valid_points_95 else 0.0

        # Calculate ROC curve for FPR-based metrics
        fpr, tpr, roc_thresholds = roc_curve(y_test, probs)

        # Calculate Recall@5% FPR
        # Find the maximum TPR (recall) where FPR <= 5%
        fpr_threshold = 0.05
        valid_tpr = tpr[fpr <= fpr_threshold]
        recall_at_5fpr = valid_tpr[-1] if len(valid_tpr) > 0 else 0.0

        # Calculate Predictive Equality (FPR Balance)
        # FPR for class 0 (majority) vs class 1 (minority)
        # For binary classification, this measures FPR consistency
        # We'll compute FPR at the operating point of 0.5 threshold
        tn = np.sum((y_test == 0) & (pred == 0))
        fp = np.sum((y_test == 0) & (pred == 1))
        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Calculate Lift@Top-K
        # Lift measures how much better the model is compared to random selection
        # Lift = (% of frauds in top K) / (% of frauds in entire dataset)
        k = int(len(y_test) * top_k_percent)
        if k == 0:
            k = 1  # Ensure at least 1 sample
        
        # Sort by predicted probability (descending)
        top_k_indices = np.argsort(probs)[-k:]
        top_k_actual = y_test.iloc[top_k_indices] if isinstance(y_test, pd.Series) else y_test[top_k_indices]
        
        # Calculate lift
        fraud_rate_top_k = np.sum(top_k_actual == 1) / k
        fraud_rate_overall = np.sum(y_test == 1) / len(y_test)
        lift_at_top_k = fraud_rate_top_k / fraud_rate_overall if fraud_rate_overall > 0 else 0.0
        
        # Calculate standard metrics at threshold 0.5
        f1 = f1_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        
        # Store results
        results.append({
            'Model': model_name,
            'PR-AUC': ap,
            'Recall@Prec>=90%': rec_at_prec_90,
            'Recall@Prec>=95%': rec_at_prec_95,
            'Recall@5%FPR': recall_at_5fpr,
            f'Lift@Top-{int(top_k_percent*100)}%': lift_at_top_k,
            'FPR@0.5': fpr_value,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df