#Oasis Infobytes_Data Science Internship
#Task 5 : SALES PREDICTION USING PYTHON
#Name of Intern : APU MANDAL
#Batch  : SEPTEMBER Phase 1 AICTE OIB-SIP 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dt = pd.read_csv("Adv.csv")
print(dt)
print("Shape of Data:",dt.shape)
print("Structure of Data:")
print(dt.describe())

dt.drop(columns=['Unnamed: 0'], axis= 1, inplace = True)
dt.isnull().sum()
dt.duplicated().sum()

sns.scatterplot(data=dt,x="TV",y="Sales",hue="Newspaper")
plt.show()

sns.scatterplot(data=dt,x="Radio",y="Sales",hue="Newspaper")
plt.show()

sns.scatterplot(data=dt,x="Newspaper",y="Sales",hue="TV")
plt.show()

x = dt.drop(['Sales'],axis=1)

y = dt['Sales']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

from sklearn.linear_model import LinearRegression

model= LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

coefficient = model.coef_
print("Coefficient od The Model:",coefficient)

intercept = model.intercept_
print("Intercept of the Model:",intercept)

from sklearn.metrics import r2_score
rs=r2_score(y_test, y_pred)*100
print("R_squared value of Model:",rs)

print("T H A N K   Y O U")
