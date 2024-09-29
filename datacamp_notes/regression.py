#Part of DAtacamp notes

#Fiiting a regression model using SciKIt Learn

import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(x_train, y)

prediction_space = np.linspace()


#Linear regression on all features

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
reg.score(x_test, y_test)
