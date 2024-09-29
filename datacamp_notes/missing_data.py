#Missing values can create bias towards the model if it is not handled properly

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values = 'Nan', strategy = 'mean', axis=0)
imp.fit(X)
X = imp.transform(X)


#Imputing within a pipeline

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

imp = Imputer(missing_values = 'Nan', strategy = 'mean', axis=0)
logreg = LogisticRegression()

steps = [('imputation', imp), ('logistic_regression', logreg)]

pipeline = Pipeline(steps)

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
pipeline.score(x_test, y_test)