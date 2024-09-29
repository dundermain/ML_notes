#cross validation notes 

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


reg = LinearRegression()
cv_results = cross_val_score(reg, x, y, cv = 5)
print(cv_results)
