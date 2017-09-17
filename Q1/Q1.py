
# coding: utf-8

# # Training Happy Singh's Sukimo bot
# 

# Happy Singh has noted all weight and diameter of apples and oranges in a csv file. This file comprise our dataset. We need to load this dataset into our program
# But how do we read the file in python? The pandas package provides function read_csv for this purpose.
# In order to use this function. we must import it our code

# In[ ]:

from pandas import read_csv


# To use the data in code, it must be stored in a variable. The NUMerical PYthon (abbv. as numpy) provides an efficient way of storing large data in objects called numpy arrays. Unlike pandas, entire numpy pakage must be imported. The type of variable that read_CSV returns, is known to numpy only

# In[ ]:

import numpy as np


# In[ ]:

apple_orange = read_csv('apple-orange-dataset.csv')


# Let's glance through our data

# In[ ]:

apple_orange


# In[ ]:

apple_orange.head()


# ### Extracting labels

# In[ ]:

y = apple_orange['label']


# In[ ]:

y.head()


# ### Extracting Features 
# Features weight and diameter, can be stored in a 2D array or matrix say X

# In[ ]:

features = ['weight','diameter']
X = apple_orange[features]


# In[ ]:

X.head()


# # Visualizing the data
# We can plot a graph of apples and oranges in 2D plane. For this we need apples and oranges in separate variables

# If you understood how apple_orange[['column_name']] works, its worth to know the following trick.

# In[ ]:

apple_orange[apple_orange['label'] == 1]


# In[ ]:

apples = apple_orange[apple_orange['label'] == 1 ]


# Feeling confident? Go ahead and do the same for oranges.

# In[ ]:

oranges = apple_orange[apple_orange['label'] == 0 ]


# The best way to find patterns in data is by plotting its graph. Here comes matplotlib package. It provides a fn plot(x,y) in a subpackage pyplot.Lets import it.

# In[ ]:

import matplotlib.pyplot as plt


# In[ ]:

plt.plot(apples['weight'],apples['diameter'], 'rs')
plt.plot(oranges['weight'],oranges['diameter'],'bs')


# Weight on X-axis and diamter on Y-axis.
# Use plt.show() to view the generated graph

# In[ ]:

plt.show()


# # Selecting a Classifier
# In order to classify apples and oranges we need to a classifier. We learnt about K-Nearest Neighbour classifier in the seminar.

# In[ ]:

from sklearn.neighbors import KNeighborsClassifier


# In[ ]:

model = KNeighborsClassifier(n_neighbors=4)


# # Train Test Split

# Our dataset is sufficiently large. So we will split the dataset into training data (say X_train, y_train) and testing data (say X_test, y_test) To ease up the task we can use the train_test_split() function from sklearn's cross_validation subpackage.

# In[ ]:

from sklearn.model_selection import train_test_split


# train_test_split is a special function. It can return multiple arrays at a time! Along with arrays, it takes test_size as argument.

# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state =1)


# X_train and y_train now contains 60% of the X and y while X_test and y_test 40% since we used test_size=0.4
# 
# Let's generate the model with fit() function and predict labels of X_test with predict()

# In[ ]:

model.fit(X_train, y_train)


# In[ ]:

y_pred = model.predict(X_test)


# In[ ]:

y_pred


# In[ ]:

list(y_test)


# To check accuracy of predictions, we will use accuracy_score fn from metrics subpackage in sklearn

# In[ ]:

from sklearn.metrics import accuracy_score


# In[ ]:

accuracy_score(y_pred, y_test)

