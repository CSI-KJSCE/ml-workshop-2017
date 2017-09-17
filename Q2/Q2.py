
# coding: utf-8

# # Steps prediction
# 

# About fitbit dataset:
#     Fitbit Inc. is an American company known for its products of the same name, which are activity trackers, 
#     wireless-enabled wearable technology devices that measure data such as the number of steps walked, heart rate, 
#     quality of sleep, steps climbed, and other personal metrics. The first of these was the Fitbit Tracker.

# For these example fitbit data of CSI member(Vivek rai :P) was collected. Here Machine learning is used to determine number of steps he walked
# (Count) based on Calories burned and Distance covered
# Since we want to "PREDICT" a missing attribute Regression is used for this example

# ### import libraries
# Import all the required libraries at once

# In[ ]:

import numpy as np
import pandas as p
import matplotlib.pyplot as plt


# ### Read CSV File (Containing fitbit dataset)

# In[ ]:

fitbit = p.read_csv("fitbit_dataset.csv")


# In[ ]:

fitbit.head()


# ### Select Input and Output features for our that dataSet (Value of X input and y output)
# 

# In[ ]:

#Here we want to predict number of steps(Count) based on Calories consumed and distance covered.
features = ['count','distance','speed']
X = fitbit[features]
y = fitbit['calorie']


# ### Split our dataset into training set and testing set
# train_test_split is a predefined function used to split data randomly
# It takes Input data to be splited along with output data
# Test size If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
# If int, represents the absolute number of test samples. 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=5)


# ### Choose apt model and create an instance of that model
# 

# In[ ]:

#In regression we are using linear regression model for prediction purposes.

from sklearn.linear_model import LinearRegression
fit_ln_model = LinearRegression()


# ### Fit the model

# In[ ]:

#Fit function will "fit" a "just fit" curve(or line) for your dataset which is apt for making prediction
#Note that fit at a certain extent will take care of overfitting  and underfitting but won't assure a right curve
# in case where data is small or ambiguous

fit_ln_model.fit(X_train,y_train)


# In[ ]:

# to check intercept and weights associated with feature use

print(fit_ln_model.intercept_)
print(fit_ln_model.coef_)


# ## Predict on test data

# In[ ]:

ypred = fit_ln_model.predict(X_test)


# Notice the output values after running next two commands

# In[ ]:

print(ypred)


# In[ ]:

print(y_test)


# ### Making a prediction for random value of calories and distance

# In[ ]:

i = [[3,27,1.6]]
test = fit_ln_model.predict(i)
print(test)


# ## Accuracy of prediction

# In[ ]:

#Accuracy is determined using predefined function of explained_variance_score

from sklearn.metrics import explained_variance_score
100*explained_variance_score(ypred,y_test)

