#Part of DAtacamp notes from the coursee ML with Scikit learn

#Classifier models = KNN



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
#staritfy splits the data into train and test splits such that train and test claass has same proportion of class labels as input dataset.


knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)