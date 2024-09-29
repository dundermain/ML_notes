#Gridsearch is way to test several values of hyperparameters inoder to find the best hyperparameter

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

param_grid = {'n_neighbours': np.arange(1,50)}

knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)

knn_cv.fit(x, y)

knn_cv.best_params_
knn_cv.best_score_