from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Prepare train/test data (already split as X_top_train, X_top_test, y_top_train, y_top_test)

# SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_top_train, y_top_train)
y_svm_pred = svm.predict(X_top_test)
svm_acc = accuracy_score(y_top_test, y_svm_pred)
print("SVM accuracy:", svm_acc)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_top_train, list(map(lambda x: 1 if x == 'ASD' else 0, y_top_train)))
y_xgb_pred = xgb.predict(X_top_test)
xgb_acc = accuracy_score(list(map(lambda x: 1 if x == 'ASD' else 0, y_top_test)), y_xgb_pred)
print("XGBoost accuracy:", xgb_acc)

# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_top_train, y_top_train)
y_lr_pred = lr.predict(X_top_test)
lr_acc = accuracy_score(y_top_test, y_lr_pred)
print("Logistic Regression accuracy:", lr_acc)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_top_train, y_top_train)
y_nb_pred = nb.predict(X_top_test)
nb_acc = accuracy_score(y_top_test, y_nb_pred)
print("Naive Bayes accuracy:", nb_acc)
