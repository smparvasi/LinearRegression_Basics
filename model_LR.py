%% Linear Regression - The simplest Possible Form

#%% Part I - Importing Modules and Dataset
#!pip install --upgrade pip --user
#!pip install seaborn
#!pip install tensorflow

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# read data
df = pd.read_csv('./Dataset/salary.csv')
df.head()

print("Maximum salary is: ",df["Salary"].max())
print("Minimum salary is: ",df["Salary"].min())

# %% Part-II Exploratory Data Analysis
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = "Blues")

df.info()
df.describe()

max = df[df["Salary"] == df["Salary"].max()]
min = df[df["Salary"] == df["Salary"].min()]

df.hist(bins = 30)

sns.pairplot(df)

plt.figure()
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()

plt.figure()
sns.regplot(x = 'YearsExperience', y = 'Salary', data = df)

# %% Create Train and Test Datasets:

X = df[['YearsExperience']]
y = df[['Salary']]

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# %% Train the model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(acc)

print(model.coef_[0])
print(model.intercept_)

# %% Model Evaluation

y_pred = model.predict(X_test)
y_pred

plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')

model.predict([[5]])

# %% 
