#ROC stands for reciever operating curve where the x = false positive rate and y = true positive rate

#for logisitic regression when p is 0.5 the ROC has the balanced true positive and false positive rate

from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

logreg = LogisticRegression()

y_pred_prob = logreg.predict_proba(x_test)[:.1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label = "Logistic Regression")
