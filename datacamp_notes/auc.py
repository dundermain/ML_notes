#AUC is the area under curve for ROC
#More the AUC better the model

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

logreg.fit(x_train, y_train)

y_pred = logreg.predict_proba(x_test)[:, 1]

roc_auc_score(y_test, y_pred)

#AUC can also be used for cross validation scoring

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(logreg, x, y, cv = 5, scoring='roc_auc')

print(cv_scores)