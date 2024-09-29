#linear regression uses loss function to find the optimum coefficient for the features
#There will be a coeffiecient for each of the feature variable. This coefficient is denoted with "a"
#Large coefficent or large "a" can lead to overfitting. So we need a way to control this. Which is why we will penalise large "a" using ridge regression


#In ridge regression we modify the loss function from simple OLS to OLS+ (alpha* a^2). Here alpha is hyperparameter and a^2 is square of each coefficient

#alpha = 0 is overfitting and high alpha will lead to underfitting

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

ridge = Ridge(alpha=0.1, normalize = True)
ridge.fit(x_train, y_train)

ridge_pred = ridge.predict(x_test)
ridge.score(x_test, y_test)