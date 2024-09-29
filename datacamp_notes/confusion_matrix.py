#confusion matrix helps in evaluating the performance of our model
#It has true positive(tp) , true negative(tn), false positive(fp) and false negative(fn)

#Accuracy is defined as (tp + tn)/(tp + tn +fp +fn) 

#precison is tp/ (tp + fp)

#recall is tp / (tp + fn)

#f1 score is 2*(precision * recall/(precision + recall))


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=8)
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))