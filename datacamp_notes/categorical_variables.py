#Categorical values cannot be used as such for machine learning models
#We have to convert it to a number

#We can do this by Scikit-learn: OneHotEncoder or Pandas:get_dummies

import pandas as pd

df = pd.read_csv('xxx.csv')
df_dummies = pd.get_dummies(df)