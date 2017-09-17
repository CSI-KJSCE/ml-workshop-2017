# -*- coding: utf-8 -*-
"""
Suppose the passing criteria for a PCM subjects exam is not known
You are given PCM score as features and passing status (pass/fail) as labels
Create an AI to figure out if a student has passed or failed without telling it the passing criteria
"""
from sklearn.neighbors import KNeighborsClassifier
# Collecting features and labels
#features : [Physics, Chemistry, Maths]
features = [[41,76,90],[35,26,51],[31,65,83],[45,65,73],[34,59,61],[43,51,36],[75,33,47],[65,90,88],[54,69,33],[65,74,35],[29,31,19],[60,62,63],[35,35,34],[35,36,40],[35,35,35]]
#labels, 1 => PASS, 0 => FAIL 
labels = [1,0,0,1,0,1,0,1,0,1,0,1,0,1,1]

#training phase
model = KNeighborsClassifier() #Instantiate empty model
model.fit(features,labels)		 # Train the model

#predict/test
print(model.predict([[75,84,59],[56,33,78]]))


