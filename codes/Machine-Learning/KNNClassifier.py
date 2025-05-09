import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from FeatureExtraction import load_graphs_from_folder, extract_graph_features


def load_and_preprocess_data(control_path, asd_path):
    """
    Load brain graphs from folders and extract features and labels.

    Parameters:
        control_path (str): Path to control group network files.
        asd_path (str): Path to ASD group network files.

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Corresponding labels.
    """
    control = load_graphs_from_folder(control_path, label=0)
    asd = load_graphs_from_folder(asd_path, label=1)

    all_graphs = control + asd
    np.random.shuffle(all_graphs)

    features = []
    labels = []

    for fALFF, edges, label in all_graphs:
        feats = extract_graph_features(fALFF, edges)
        features.append(feats)
        labels.append(label)

    X = pd.DataFrame(features)
    X = X.fillna(X.mean())  # Handle missing values
    y = np.array(labels)

    return X, y


def evaluate_knn(X, y, n_neighbors=5):
    """
    Train and evaluate a KNN model using cross-validation and full training set.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Label array.
        n_neighbors (int): Number of neighbors for KNN.

    Returns:
        None
    """
    knn_model = KNeighborsClassifier(n_neighbors=5)

    # Perform 5-fold cross-validation
    cv_scores_knn = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')

    print("Cross-validation scores: ", cv_scores_knn)
    print("Mean accuracy: ", cv_scores_knn.mean())
    print("Standard deviation of accuracy: ", cv_scores_knn.std())

    # Fit model on full data for final classification report
    knn_model.fit(X, y)

    y_pred_knn = knn_model.predict(X)

    print("Classification Report:\n", classification_report(y, y_pred_knn))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred_knn))


if __name__ == "__main__":
    # Paths to the datasets
    control_folder = r'directory-to-control-data'
    asd_folder = r'directory-to-asd-data'

    # Load data and run KNN classifier
    X, y = load_and_preprocess_data(control_folder, asd_folder)
    evaluate_knn(X, y)
