import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (cross_val_score, GridSearchCV,
                                   StratifiedKFold)
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                           classification_report, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
import seaborn as sns

from NetworkEnergyComputer import compute_network_energy
from SubNetworkFeatureExtraction import subnetwork_energy_features, compute_subnetwork_energy
from FeatureExtraction import load_graphs_from_folder, extract_graph_features



def load_and_preprocess_data(asd_dir, ctrl_dir):


    # Load and preprocess data
    control = load_graphs_from_folder(ctrl_dir, label=0)
    asd = load_graphs_from_folder(asd_dir, label=1)
    all_graphs = control + asd
    np.random.shuffle(all_graphs)

    # Feature extraction
    features, labels = [], []
    for fALFF, edges, label, path in all_graphs:
        feats = extract_graph_features(fALFF, edges)
        energies, _ = subnetwork_energy_features(path, label)
        feats.update({f"energy_{i}": e for i, e in enumerate(energies)})
        features.append(feats)
        labels.append(label)

    X = pd.DataFrame(features).fillna(0)
    y = np.array(labels)

    return X,y



def evaluate_svm(X, y, SEED):


    # ======================== MODEL EVALUATION ========================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Baseline model
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr_scores = cross_val_score(lr, X, y, cv=cv)
    print(f"Logistic Regression CV: {lr_scores.mean():.3f} (±{lr_scores.std():.3f})")

    # SVM with tuning
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True, random_state=SEED)
    )

    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }

    search = GridSearchCV(svm, param_grid, cv=cv, n_jobs=-1, verbose=1)
    search.fit(X, y)
    best_model = search.best_estimator_

    # Generate predictions for ROC curve
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    # ======================== VISUALIZATION ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(
        best_model, X, y, ax=ax1, normalize='true',
        display_labels=['Control', 'ASD'], cmap='Blues'
    )
    ax1.set_title(f"SVM Confusion Matrix\n(Accuracy: {search.best_score_:.2f})")

    # ROC Curve
    RocCurveDisplay.from_estimator(
        best_model, X, y, ax=ax2
    )
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    ax2.set_title(f"SVM ROC Curve (AUC = {roc_auc:.2f})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('results.png', bbox_inches='tight')
    plt.show()

    # ======================== STATISTICAL REPORT ======================
    print("\n=== Best Model ===")
    print(f"Parameters: {search.best_params_}")
    print(f"CV Accuracy: {search.best_score_:.3f} (±{cross_val_score(best_model, X, y, cv=cv).std():.3f})")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Control', 'ASD']))



if __name__ == "__main__":

    # Reproducibility Control

    SEED = 42
    np.random.seed(SEED)
    sns.set_theme(style="whitegrid")  # Modern Seaborn theming
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'Arial'

    # Evaluation

    asd_dir = r"D:\University\projects\CBTProject\CoevolutionaryBalanceTheory\Data\ASDNetworks"
    ctrl_dir = r"D:\University\projects\CBTProject\CoevolutionaryBalanceTheory\Data\ControlNetworks"
    X, y = load_and_preprocess_data(asd_dir, ctrl_dir)
    evaluate_svm(X, y, SEED)