
# coding: utf-8

# # Types of drivers in USA

# ### Import libraries

# In[1]:

import numpy as np
import pandas as p
import matplotlib.pyplot as plt


# ### Load data from csv file

# In[4]:

dataset = p.read_csv('speeding_feature_data.csv')


# ### Extract features from data
# First see how the data looks like

# In[61]:

dataset.head()


# In[62]:

dataset.tail()


# We saw there are only two features in our dataset

# In[8]:

features = ['Distance_Feature','Speeding_Feature']
x = dataset[features]


# ### Split the data into two parts
# First import the train_test_split function from sklearn library

# In[63]:

from sklearn.model_selection import train_test_split


# In[64]:

X_train, X_test = train_test_split(x, test_size=0.5,random_state=5)


# ### Select a model
# Since this is an unsupervised clustering problem, we will use KMeans algorithm to train our model. We want four types of drivers, so we will speicfy n_clusters=4

# In[82]:

from sklearn.cluster import KMeans


# In[50]:

model = KMeans(n_clusters=4)


# ### Train the model with fit function
# In unsupervised learning, we do not give labels but only features and let the model form clusters of data

# In[68]:

model.fit(X_train)


# ### View labels returned by our model
# KMeans will assign labels to the clusters. These labels are stores in its labels_  attribute

# In[74]:

labels = model.labels_
labels


# ### Get unique labels
# So we will use numpy.unique() function that will give us only unique labels

# In[75]:

np.unique(labels)


# ### Test the model

# In[77]:

y_pred = model.predict(X_test)


# In[81]:

y_pred

