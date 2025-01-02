# %% [markdown]
# # **Solar Power Generation**

# %%


# %% [markdown]
# ### Importing required libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%


# %% [markdown]
# ### Reading the Dataset

# %%
df=pd.read_csv(r"/Users/abhi/Downloads/solarpowergeneration.csv")
df

# %%


# %% [markdown]
# ### checking the shape of the Dataset

# %%
df.shape


# %%


# %% [markdown]
# ### Displaying the first and last rows of a dataset

# %%
df.head()

# %%
df.tail()

# %%


# %% [markdown]
# ### Checking the data types of each column

# %%
df.dtypes

# %%
df.info()

# %%


# %% [markdown]
# ### Names of all the columns in the Dataset

# %%
df.columns

# %%


# %% [markdown]
# ### For checking null values in the Dataset

# %%
df.isnull().sum()

# %%


# %% [markdown]
# ### For checking duplicant values

# %%
df[df.duplicated()]

# %%


# %% [markdown]
# ### Identifying Rows with Missing Values

# %%
df[df.isnull().any(axis=1)]


# %%


# %% [markdown]
# ### Counting Missing Values in a Specific Column

# %%
df['average-wind-speed-(period)'].isnull().sum()

# %%


# %% [markdown]
# ### Filling Missing Values in a Column with Zeros

# %%
df['average-wind-speed-(period)'] = df['average-wind-speed-(period)'].fillna(0.0)


# %%
df.isnull().sum()

# %%


# %% [markdown]
# ### Descriptive statistics

# %%
df.describe()

# %%


# %% [markdown]
# ### Correlation Analysis

# %%
df.corr()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score


# %%
df = pd.read_csv("solarpowergeneration.csv")

# %%
df

# %%
df.describe()

# %%
df.info()

# %%
temp=df.drop(df.columns[[3,4,8]],axis=1)

# %%
temp.corr()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing

# %%
solarpowergeneration = pd.read_csv('solarpowergeneration.csv')

# %%
solarpowergeneration.head()

# %%
x=solarpowergeneration.iloc[:,0:8]
y=solarpowergeneration['power-generated']

# %%
x

# %%
y

# %%
solarpowergeneration['power-generated'].unique()

# %%
colnames = list(solarpowergeneration.columns)
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
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.tree import DecisionTreeRegressor

# %%
array = solarpowergeneration.values
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



