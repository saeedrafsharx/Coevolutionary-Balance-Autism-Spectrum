import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('graph_features_full.csv')
y = df['label']
X = df.drop(columns=['label'])
# 1. Fit Random Forest to find best features
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(9).index.tolist()
print("Top 8 features:", top_features)
# Use only the top features for KNN
X_top = df[top_features]
X_top = StandardScaler().fit_transform(X_top)
X_top_train, X_top_test, y_top_train, y_top_test = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)


# Fit KNN with best K
knn_top = KNeighborsClassifier(n_neighbors=11)
knn_top.fit(X_top_train, y_top_train)

# Predict and evaluate
y_top_pred = knn_top.predict(X_top_test)
print("Accuracy (top features):", accuracy_score(y_top_test, y_top_pred))
print(classification_report(y_top_test, y_top_pred))
ConfusionMatrixDisplay.from_predictions(y_top_test, y_top_pred, display_labels=['ASD', 'TD'], cmap='Purples', colorbar=False)
plt.title("Confusion Matrix (KNN, Top Features)")
plt.show()
