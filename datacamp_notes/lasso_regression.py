#In lasso regression we modify the loss function from simple OLS to OLS+ (alpha* a). Here alpha is hyperparameter and a is absolute value of each coefficient

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

lasso = Lasso(alpha=0.1, normalize = True)
lasso.fit(x_train, y_train)

lasso_pred = lasso.predict(x_test)
lasso.score(x_test, y_test)


#One cool feature of lasso regression is that it can be used for feature selection.
#lasso reduces the coefficient of less important features to zero
#coefficient represents how much value or weight should be assigned for features depeendng on their importance/contribution in predicition or regression
#hence a = 0 means no contribution by that feature