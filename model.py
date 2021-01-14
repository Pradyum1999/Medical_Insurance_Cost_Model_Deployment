#!/usr/bin/env python
# coding: utf-8

# In[197]:


import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("insurance.csv")

lb = LabelEncoder()
data['sex'] = lb.fit_transform(data.sex)

lb = LabelEncoder()
data['smoker'] = lb.fit_transform(data.smoker)

lb = LabelEncoder()
data['region'] = lb.fit_transform(data.region)

X = data.iloc[:,0:6]
y = data.iloc[:,6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=25)

lr = Lasso(alpha= 0.6, max_iter=5000, random_state=25)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

import pickle

pickle.dump(lr, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[21, 1, 27, 0, 0, 2]]))

