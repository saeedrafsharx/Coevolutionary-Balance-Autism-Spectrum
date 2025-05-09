import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, roc_curve, auc)
import seaborn as sns


from codes.NetworkEnergyComputer import compute_network_energy
from SubNetworkFeatureExtraction import subnetwork_energy_features, compute_subnetwork_energy
from FeatureExtraction import load_graphs_from_folder, extract_graph_features


def load_and_preprocess_data(asd_dir, ctrl_dir):
    """
    Load brain graphs from folders and extract features and labels.

    Parameters:
        control_path (str): Path to control group network files.
        asd_path (str): Path to ASD group network files.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Corresponding labels.
    """
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

    return X, y


def evaluate_knn(X, y, SEED):


    # Model Evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # KNN with feature scaling
    knn_pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier()
    )

    # Hyperparameter grid
    param_grid = {
        'kneighborsclassifier__n_neighbors': list(range(3, 15, 2)),  # Test odd numbers
        'kneighborsclassifier__weights': ['uniform', 'distance'],
        'kneighborsclassifier__p': [1, 2],  # Manhattan vs Euclidean
        'kneighborsclassifier__metric': ['minkowski', 'cosine']  # Additional metric
    }

    # Grid search with parallel processing
    knn_search = GridSearchCV(knn_pipe, param_grid, cv=cv, n_jobs=-1, verbose=1)
    knn_search.fit(X, y)
    best_knn = knn_search.best_estimator_

    # Generate predictions
    y_pred = best_knn.predict(X)
    y_prob = best_knn.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix (Normalized)
    ConfusionMatrixDisplay.from_estimator(
        best_knn, X, y, ax=ax1, normalize='true',
        display_labels=['Control', 'ASD'], cmap='Blues'
    )
    ax1.set_title(f"KNN Confusion Matrix\n(Accuracy: {knn_search.best_score_:.2f})")

    # ROC Curve
    RocCurveDisplay.from_estimator(best_knn, X, y, ax=ax2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)')
    ax2.set_title(f"KNN ROC Curve\n(AUC = {roc_auc:.2f})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('knn_performance.png', bbox_inches='tight', dpi=300)
    plt.show()

    # Statistical Report
    print("\n=== Optimal KNN Model ===")
    print(f"Best Parameters: {knn_search.best_params_}")
    print(f"CV Accuracy: {knn_search.best_score_:.3f} (±{cross_val_score(best_knn, X, y, cv=cv).std():.3f})")

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, target_names=['Control', 'ASD']))

    # Feature Importance (Permutation Importance)
    from sklearn.inspection import permutation_importance

    print("\nCalculating feature importance...")
    result = permutation_importance(best_knn, X, y, n_repeats=10, random_state=SEED)
    important_features = pd.Series(result.importances_mean, index=X.columns)
    print("\nTop 10 Features:")
    print(important_features.sort_values(ascending=False).head(10))


if __name__ == "__main__":
    # Reproducibility Control
    SEED = 42
    np.random.seed(SEED)
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
    plt.rcParams.update({'font.family': 'Arial', 'figure.dpi': 300})

    # Evaluation
    asd_dir = r'directory-to-control-data'
    ctrl_dir = r'directory-to-asd-data'
  
    X, y = load_and_preprocess_data(asd_dir, ctrl_dir)
    evaluate_knn(X, y, SEED)
