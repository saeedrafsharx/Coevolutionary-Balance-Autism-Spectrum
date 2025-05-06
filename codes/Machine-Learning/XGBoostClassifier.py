import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from FeatureExtraction import load_graphs_from_folder, extract_graph_features


def load_and_preprocess_data(control_path, asd_path):
    """
    Load brain graph data, extract features, and prepare labels.

    Parameters:
        control_path (str): Path to control group folder.
        asd_path (str): Path to ASD group folder.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Labels.
    """
    control = load_graphs_from_folder('Data/ControlNetworks', label=0)
    asd = load_graphs_from_folder('Data/ASDNetworks', label=1)

    all_graphs = control + asd
    # %%
    feature_dicts = [extract_graph_features(fALFF, edges) for fALFF, edges, label in all_graphs]
    labels = [label for _, _, label in all_graphs]

    X = pd.DataFrame(feature_dicts)
    y = np.array(labels)

    X = X.fillna(X.mean())

    return X, y



def evaluate_xgboost(X, y):
    """
    Train and evaluate an XGBoost classifier.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Label array.

    Returns:
        None
    """
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # ROC curve
    y_pred = cross_val_predict(model, X, y, cv=cv)
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    control_folder = 'Data/ControlNetworks'
    asd_folder = 'Data/ASDNetworks'

    X, y = load_and_preprocess_data(control_folder, asd_folder)
    evaluate_xgboost(X, y)
