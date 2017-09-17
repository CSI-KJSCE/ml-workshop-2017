
# coding: utf-8

# # Titanic survivals
# The world remembers the Titanic disaster where many people lost their lives. But there are many others who survived the disaster. We have collected data about a lot of passengers in an excel file (titanic.xls). Suppose you were one of the passengers in the ship.Your Port of Embarkation was 'S', ticket fare 200, Passenger Class was 3 and your sex and age were your actual sex and current age. What could have been your chances of survival?

# In[1]:

# Description of columns in the excel data
'''
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)
'''


# Import required libraries and functions

# In[2]:

import pandas as p
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ### Load dataset
# Our dataset is in .xls format, google to findout which function in pandas can be used to load .xls data

# In[3]:

df = p.read_excel('titanic.xls') # load dataset using pandas. eg. p.some_function('titanic.xls')


# ### Extract useful features
# View first 5 rows to figure out which features can be used for prediction

# In[4]:

#To see a portion of your dataset use head() function
df.head()


# In[5]:

df.describe()


# In[6]:

#Notice that count of age is less so fillna is used to fill all the null values in dataset with median.
#So is the case of embarked.
df["age"]= df["age"].fillna(df["age"].median())
df["embarked"]=df["embarked"].fillna("S")
df["fare"]=df["fare"].fillna(df["fare"].median())


# In[7]:

#Sex and embarked are factors but we can notice that values are string 
#in order to use them as features we have to convert them into integer

df.loc[df["sex"]=="male","sex"]=0
df.loc[df["sex"]=="female","sex"]=1

df.loc[df["embarked"]=="S","embarked"]=0
df.loc[df["embarked"]=="C","embarked"]=1
df.loc[df["embarked"]=="Q","embarked"]=2


# In[8]:

features = ['sex','age',"fare", "embarked", "pclass"]
X = df[features]
Y = df["survived"]


# ### Split features and labels into training and testing data

# In[9]:

#Input and output data stored in X and y is splitted into training and testing dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4,random_state=2)


# ### Select and train your model

# In[10]:

#Create an instance of KNeighborsClassifier with parameter n_neighbors=3

knn = KNeighborsClassifier(n_neighbors=3)


# In[11]:

#train your model with fit() method
knn.fit(X_train,Y_train)


# ### Test your model

# In[12]:

# Predict/Classify value of output for X_test
dfResult = knn.predict(X_test)


# In[13]:

print(dfResult)
print("Our accuracy : ", accuracy_score(dfResult, Y_test))


# In[14]:

print("To test for best value of K and plotting the graph ")
acc = []
best_k = 1
best_accuracy = 0
for k in range(1,36):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    acc.append(accuracy_score(knn.predict(X_test), Y_test))
    if(acc[k-1] > best_accuracy):
        best_accuracy = acc[k-1]
        best_k = k
plt.plot(range(1,36), acc)
plt.xlabel("Value of K")
plt.ylabel("Accuracy ")
plt.show()
print("Best value for k is "+str(best_k))
print("Highest accuracy is "+str(best_accuracy))


# ### Train the model using best value of k obtained above

# In[15]:

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, Y_train)


# ### Check your chances of survival

# In[16]:

knn.predict([[0,22,200,0,3]])

