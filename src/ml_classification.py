"""
Machine Learning Classification: ASD vs TD

Leakage-safe pipeline with multiple classifiers for autism classification
using coevolutionary network features.

Usage:
    python ml_classification.py --input features.csv --output_dir results/
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                            confusion_matrix, roc_curve, auc, classification_report)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    return np.percentile(scores, [alpha*100, (1-alpha)*100])


def select_features(X_train, y_train, top_frac=0.25, min_feat=10):
    """Select top features using Random Forest importance."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    n_select = max(min_feat, int(len(X_train.columns) * top_frac))
    selected = importance.head(n_select)['Feature'].tolist()
    
    return selected, importance


def get_models():
    """Define model configurations."""
    models = {
        'GaussianNB': {
            'pipe': Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]),
            'params': {'clf__var_smoothing': [1e-9, 1e-8, 1e-7]}
        },
        'SVM': {
            'pipe': Pipeline([('scaler', StandardScaler()), 
                             ('clf', SVC(probability=True, class_weight='balanced', random_state=42))]),
            'params': {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 0.01, 0.001]}
        },
        'LogisticRegression': {
            'pipe': Pipeline([('scaler', StandardScaler()),
                             ('clf', LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42))]),
            'params': {'clf__C': [0.01, 0.1, 1, 10], 'clf__penalty': ['l2']}
        },
        'KNN': {
            'pipe': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
            'params': {'clf__n_neighbors': [5, 9, 15, 21]}
        }
    }
    
    if HAS_XGB:
        models['XGBoost'] = {
            'pipe': Pipeline([('scaler', StandardScaler()),
                             ('clf', XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0))]),
            'params': {'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 5], 
                      'clf__learning_rate': [0.01, 0.1]}
        }
    
    return models


def train_evaluate(X_train, X_test, y_train, y_test, output_dir):
    """Train all models and evaluate on test set."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = get_models()
    results = []
    best_model = None
    best_acc = 0
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        grid = GridSearchCV(config['pipe'], config['params'], cv=cv, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        acc_ci = bootstrap_ci(y_test, y_pred, accuracy_score)
        bal_ci = bootstrap_ci(y_test, y_pred, balanced_accuracy_score)
        f1_ci = bootstrap_ci(y_test, y_pred, lambda yt, yp: f1_score(yt, yp, average='macro'))
        
        results.append({
            'model': name,
            'best_params': json.dumps(grid.best_params_),
            'cv_score': grid.best_score_,
            'test_acc': acc,
            'test_bal_acc': bal_acc,
            'test_f1_macro': f1,
            'acc_ci_low': acc_ci[0],
            'acc_ci_high': acc_ci[1],
            'bal_acc_ci_low': bal_ci[0],
            'bal_acc_ci_high': bal_ci[1],
            'f1_ci_low': f1_ci[0],
            'f1_ci_high': f1_ci[1]
        })
        
        print(f"  CV: {grid.best_score_:.3f}, Test: {acc:.3f} [{acc_ci[0]:.3f}-{acc_ci[1]:.3f}]")
        
        if acc > best_acc:
            best_acc = acc
            best_model = (name, grid.best_estimator_, y_pred, y_proba)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/final_results.csv', index=False)
    
    # Plot best model
    if best_model:
        name, model, y_pred, y_proba = best_model
        plot_results(y_test, y_pred, y_proba, name, output_dir)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba if y_proba is not None else np.nan
        })
        pred_df.to_csv(f'{output_dir}/best_model_test_predictions.csv', index=False)
    
    # Plot comparison
    plot_comparison(results_df, output_dir)
    
    return results_df


def plot_results(y_true, y_pred, y_proba, model_name, output_dir):
    """Plot confusion matrix and ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
               xticklabels=['ASD', 'Control'], yticklabels=['ASD', 'Control'])
    axes[0].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')
    axes[0].set_title(f'Confusion Matrix ({model_name})')
    
    # ROC curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, 'crimson', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0,1], [0,1], 'gray', lw=2, linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve ({model_name})')
        axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/best_model_{model_name}_confusion_auc.pdf', dpi=150)
    plt.close()


def plot_comparison(results_df, output_dir):
    """Bar plot comparing model accuracies."""
    df = results_df.sort_values('test_acc', ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['model'], df['test_acc'], color='steelblue', edgecolor='black')
    
    for bar, acc in zip(bars, df['test_acc']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', fontsize=10)
    
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.title('Model Test Accuracy (final_results.csv)')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/final_results_barplot.pdf', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='ML Classification')
    parser.add_argument('--input', required=True, help='Features CSV')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {args.input}: {df.shape}")
    
    # Prepare features
    meta = {'Subject', 'Group', 'Center', 'FIQ', 'IQ'}
    feat_cols = [c for c in df.columns if c not in meta]
    
    X = df[feat_cols].copy()
    le = LabelEncoder()
    y = le.fit_transform(df['Group'])
    
    print(f"[INFO] Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Drop missing
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    print(f"[INFO] Samples: {len(y)}, Features: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Feature selection
    selected, importance = select_features(X_train, y_train)
    print(f"[INFO] Selected {len(selected)} features")
    
    importance.to_csv(f'{args.output_dir}/random_forest_feature_importance.csv', index=False)
    
    X_train_sel = X_train[selected].values
    X_test_sel = X_test[selected].values
    
    # Train and evaluate
    results = train_evaluate(X_train_sel, X_test_sel, y_train, y_test, args.output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
