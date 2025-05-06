import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from FeatureExtraction import load_graphs_from_folder, extract_graph_features


def load_and_preprocess_data(control_path, asd_path):
    """
    Loads graph data, extracts features, and prepares labels.

    Parameters:
        control_path (str): Path to control group data.
        asd_path (str): Path to ASD group data.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Labels.
    """
    control = load_graphs_from_folder('Data/ControlNetworks', label=0)
    asd = load_graphs_from_folder('Data/ASDNetworks', label=1)

    all_graphs = control + asd

    features = []
    labels = []
    np.random.shuffle(all_graphs)

    for fALFF, edges, label in all_graphs:
        feats = extract_graph_features(fALFF, edges)
        features.append(feats)
        labels.append(label)

    X = pd.DataFrame(features)
    y = np.array(labels)

    X = X.fillna(X.mean())

    return X, y
def evaluate_svm(X, y):
    """
    Evaluates an SVM classifier using cross-validation and ROC.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Labels.

    Returns:
        None
    """
    # Define SVM model with scaling
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

    # Cross-validation
    svm_scores = cross_val_score(svm_model, X, y, cv=5)

    print("SVM accuracy (5-fold CV):", svm_scores.mean())

    # Hyperparameter tuning
    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }


    grid_search = GridSearchCV(svm_model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)

    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)

    svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
    svm_model.fit(X, y)

    y_pred = svm_model.predict(X)
    y_prob = svm_model.predict_proba(X)[:, 1]

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


    print("Classification Report:\n", classification_report(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

if __name__ == "__main__":
    control_folder = 'Data/ControlNetworks'
    asd_folder = 'Data/ASDNetworks'

    X, y = load_and_preprocess_data(control_folder, asd_folder)
    evaluate_svm(X, y)
