#logisitic regression gives ouput in probabilites
#if p < 0.5, output is 0 otherwise 1


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

logreg.fit(x_train, y_train)
y_pred = logreg.pred(x_test)
