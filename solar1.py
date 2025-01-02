# %%
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore


# %%
df=pd.read_csv(r"C:\Users\khobr\OneDrive\Desktop\DATA Science\Projects\excel r 1 project\solarpowergeneration.csv")
df

# %%
df.shape

# %%
df.dtypes

# %%
df.info()


# %%
df.columns

# %%
df.isnull().sum()

# %%
df[df.duplicated()]

# %%
df[df.isnull().any(axis=1)]

# %%
df['average-wind-speed-(period)'].isnull().sum()

# %%
df['average-wind-speed-(period)'] = df['average-wind-speed-(period)'].fillna(0.0)

# %%
df.isnull().sum()

# %%
df.describe()

# %%
df.corr()

# %%
from sklearn.metrics import accuracy_score # type: ignore

# %%
temp=df.drop(df.columns[[3,4,8]],axis=1)

# %%
temp.corr()

# %%
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Handle missing values by filling with the median (as a simple approach)
df['average-wind-speed-(period)'].fillna(df['average-wind-speed-(period)'].median(), inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=['power-generated'])
y = df['power-generated']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2

# %%
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn import datasets   # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import  DecisionTreeClassifier # type: ignore
from sklearn import tree # type: ignore
from sklearn.metrics import classification_report # type: ignore
from sklearn import preprocessing # type: ignore

# %%
x=df.iloc[:,0:8]
y=df['power-generated']

# %%
x

# %%
y

# %%
df['power-generated'].unique()

# %%
colnames = list(df.columns)
colnames

# %%
# Splitting data into training and testing data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)

# %%
#Building Decision Tree Classifier using Entropy Criteria

# %%
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)

# %%
#PLot the decision tree
tree.plot_tree(model);

# %%
#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 

# %%
preds

# %%
pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions

# %%
# Accuracy
np.mean(preds==y_test)

# %%
#Building Decision Tree Classifier (CART) using Gini Criteria¶

# %%
from sklearn.tree import DecisionTreeClassifier # type: ignore
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)

# %%
model_gini.fit(x_train, y_train)

# %%
#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)

# %%
#Decision Tree Regression

# %%
from sklearn import DecisionTreeRegressor # type: ignore

# %%
array = df.values
X = array[:,0:8]
y = array[:,8]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)

# %%
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# %%
#Find the accuracy
model.score(X_test,y_test)

# %%



