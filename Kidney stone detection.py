#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
import dask.distributed

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/ckdisease/kidney_disease.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# In[4]:


start = time.time()
get_ipython().run_line_magic('time', 'temp  =  dd.read_csv(r"C:\\Users\\USER\\Downloads\\kidney_disease.csv")')
print("Time taken:"+str(time.time()-start))


# In[5]:


start = time.time()
get_ipython().run_line_magic('time', 'data = pd.read_csv(r"C:\\Users\\USER\\Downloads\\kidney_disease.csv")')
print("Time taken:"+str(time.time()-start))


# In[6]:


data.head()


# In[7]:


#getting the shape of the dataset
data.shape


# In[8]:


data.columns


# In[9]:


#getting the information about the dataset contents
data.info()


# In[10]:


categorial_cols = [col for col in data.columns if data[col].dtype=="object"]
categorial_cols


# In[11]:


numerical_cols = [x for x in data.columns if not x in categorial_cols]
numerical_cols


# In[12]:


for i in ['rc','wc','pcv']:
    data[i] = data[i].str.extract('(\d+)').astype(float)


# In[13]:


#filling the null values with the mean values 
for i in ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','rc','wc','pcv']:
    data[i].fillna(data[i].mean(),inplace=True)


# In[14]:


#converting the categorial data by using oneHot Encoding
rbc = pd.get_dummies(data[["rbc"]],drop_first=True)
rbc.head()


# In[15]:


pc = pd.get_dummies(data[["pc"]],drop_first=True)
pc.head()


# In[16]:


pcc = pd.get_dummies(data[["pcc"]],drop_first=True)
pcc.head()


# In[17]:


ba = pd.get_dummies(data[["ba"]],drop_first=True)
ba.head()


# In[18]:


#dropping the categorial data columns
data.drop(["rbc","pc","pcc","ba"],axis=1,inplace=True)


# In[19]:


data.head()


# In[20]:


#concating the data columns
data = pd.concat([data,rbc,pc,pcc,ba],axis=1)


# In[21]:


data.head()


# In[22]:


data.info()


# In[23]:


#converting the age Data column into list
k=data["age"].apply(lambda x : int(x)//10).to_list()


# In[24]:


#batching the ages(0-9,10-19,20-29,......90-99)
a=[0]*10
for i in range(len(k)):
    a[k[i]]+=1
a


# In[25]:


plt.figure(figsize= (7,7))
x_labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99']
y_labels = np.array(a)
plt.pie(y_labels, labels = x_labels)
plt.show()


# In[26]:


data["classification"].value_counts()


# In[27]:


#replacing the values of notckd, ckd and ckd/t in the dataset
data.replace({"notckd":0,"ckd":1,"ckd\t":1},inplace=True)


# In[28]:


plt.figure(figsize=(25,10))
sns.barplot(x =data['bp'],y =data['classification'],data=data)
plt.show()


# In[29]:


plt.figure(figsize=(10,10))
sns.swarmplot(y=data["age"], x = data["classification"])


# In[30]:


data["appet"].value_counts()


# In[31]:


plt.figure(figsize=(7,7))
sns.barplot(x =data['appet'],y =data['classification'],data=data)
plt.show()


# In[32]:


data.replace({"good":1,"poor":0},inplace=True)


# In[33]:


data["ane"].value_counts()


# In[34]:


#replacing the values of no, yes to 0,1 respectively
data.replace({"no":0,"yes":1,"\tno":0,"\tyes":1," yes":1},inplace=True)


# In[35]:


data.info()


# In[36]:


#frequent value in the column
data=data.apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[37]:


data.info()


# In[38]:


#getting the columns in the dataset
data.columns


# In[39]:


#seperating the data for the model as X contains the data which feed to the model and y contains the target column 
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]


# In[40]:


#splitting the train data and test Data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
X_train.head()


# # RANDOM FOREST CLASSIFIER 

# In[41]:


#using the random forest classifier
model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[42]:


#getting the predictions using the trained model
predictions = model.predict(X_test)


# In[43]:


#checking the mean absolute error between the predicted values and test data
print("the mean absolute error by using the RandomForest is",mean_absolute_error(y_test,predictions))


# In[44]:


#printing the accuracy of the train data
print("the accuracy of the train data is ",model.score(X_train,y_train)*100)


# In[45]:


#printing the accuracy of the test data
print("the accuracy of the test data is",model.score(X_test,y_test)*100)


# # Random forest classifier using a single core

# In[46]:


# define the model
model = RandomForestClassifier(n_estimators=500, n_jobs=1)


# In[47]:


# timing the training of a random forest model on one core
from time import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=500, n_jobs=1)
# record current time
start = time()
# fit the model
model.fit(X, y)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# # Random forest classifier using 4 cores

# In[48]:


# define the model
model = RandomForestClassifier(n_estimators=500, n_jobs=4)


# In[49]:


# example of timing the training of a random forest model on 4 cores
from time import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=500, n_jobs=4)
# record current time
start = time()
# fit the model
model.fit(X, y)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# # Utilizing all the cores (8)

# In[50]:


# define the model
model = RandomForestClassifier(n_estimators=500, n_jobs=-1)


# In[51]:


# example of timing the training of a random forest model on 8 cores
from time import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# define dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# record current time
start = time()
# fit the model
model.fit(X, y)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# In[52]:


# example of comparing number of cores used during training to execution speed
from time import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=10000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
results = list()
# compare timing for number of cores
n_cores = [1, 2, 3, 4, 5, 6, 7, 8]
for n in n_cores:
	# capture current time
	start = time()
	# define the model
	model = RandomForestClassifier(n_estimators=500, n_jobs=n)
	# fit the model
	model.fit(X, y)
	# capture current time
	end = time()
	# store execution time
	result = end - start
	print('>cores=%d: %.3f seconds' % (n, result))
	results.append(result)
pyplot.plot(n_cores, results)
pyplot.show()


# # Evaluvating our model using single core

# In[53]:


# define the model
model = RandomForestClassifier(n_estimators=100, n_jobs=1)


# In[54]:


# example of evaluating a model using a single core
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=100, n_jobs=1)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# record current time
start = time()
# evaluate the model
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=1)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# In[55]:


# example of evaluating a model using 8 cores
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=100, n_jobs=1)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# record current time
start = time()
# evaluate the model
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=8)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# In[ ]:





# # Evaluvating our model with 8 processors

# In[56]:



from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=100, n_jobs=1)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# record current time
start = time()
# evaluate the model
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=8)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# # comparing evaluvation model with single core vs 8 cores

# In[57]:


# compare execution speed for model evaluation vs number of cpu cores
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
results = list()
# compare timing for number of cores
n_cores = [1, 2, 3, 4, 5, 6, 7, 8]
for n in n_cores:
	# define the model
	model = RandomForestClassifier(n_estimators=100, n_jobs=1)
	# define the evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# record the current time
	start = time()
	# evaluate the model
	n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=n)
	# record the current time
	end = time()
	# store execution time
	result = end - start
	print('>cores=%d: %.3f seconds' % (n, result))
	results.append(result)
pyplot.plot(n_cores, results)
pyplot.show()


# # random forest classifier hyper parameter tuning

# In[58]:


# example of tuning model hyperparameters with a single core
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=100, n_jobs=1)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['max_features'] = [1, 2, 3, 4, 5]
# define grid search
search = GridSearchCV(model, grid, n_jobs=1, cv=cv)
# record current time
start = time()
# perform search
search.fit(X, y)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# In[59]:


# example of tuning model hyperparameters with 8 cores
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=100, n_jobs=1)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['max_features'] = [1, 2, 3, 4, 5]
# define grid search
search = GridSearchCV(model, grid, n_jobs=8, cv=cv)
# record current time
start = time()
# perform search
search.fit(X, y)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# In[60]:


# example of multi-core model training and hyperparameter tuning
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# define dataset
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
# define the model
model = RandomForestClassifier(n_estimators=100, n_jobs=4)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['max_features'] = [1, 2, 3, 4, 5]
# define grid search
search = GridSearchCV(model, grid, n_jobs=4, cv=cv)
# record current time
start = time()
# perform search
search.fit(X, y)
# record current time
end = time()
# report execution time
result = end - start
print('%.3f seconds' % result)


# # XGBOOST ALGORITHM

# In[61]:


#using the XGradient Boosting algorithm
mod = XGBClassifier()
mod.fit(X_train,y_train)


# In[62]:


#checking the mean absolute error between the predicted values and test data
print(mean_absolute_error(y_test,mod.predict(X_test)))


# In[63]:


#printing the accuracy of the train data
print("the accuracy of the train data is ",mod.score(X_train,y_train)*100)


# In[64]:


print("the accuracy of the test data is",mod.score(X_test,y_test)*100)


# # UTLIZING ALL THE 8 CORES

# In[66]:


model = XGBClassifier(nthread=-1)


# In[67]:


# tune number of threads
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import time
from matplotlib import pyplot
# load data
#data = read_csv('train.csv')
#dataset = data.values
# split data into X and y
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# evaluate the effect of the number of threads
results = []
num_threads = [1, 2, 3, 4]
for n in num_threads:
	start = time.time()
	model = XGBClassifier(nthread=n)
	model.fit(X, label_encoded_y)
	elapsed = time.time() - start
	print(n, elapsed)
	results.append(elapsed)
# plot results
pyplot.plot(num_threads, results)
pyplot.ylabel('Speed (seconds)')
pyplot.xlabel('Number of Threads')
pyplot.title('XGBoost Training Speed vs Number of Threads')
pyplot.show()


# # PARALLELISM WHEN CROSS VALIADATING XGBOOST MODELS

# In[68]:


# parallel cross validation
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import time

# split data into X and y
X = data.loc[:,['age', 'bp', 'rc','wc','appet','pc_normal','htn','hemo','bgr','dm','ane']]
y = data["classification"]
# encode string class values as integers
label_encoded_y = LabelEncoder().fit_transform(y)
# prepare cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# Single Thread XGBoost, Parallel Thread CV
start = time.time()
model = XGBClassifier(nthread=1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
print("Single Thread XGBoost, Parallel Thread CV: %f" % (elapsed))
# Parallel Thread XGBoost, Single Thread CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=1)
elapsed = time.time() - start
print("Parallel Thread XGBoost, Single Thread CV: %f" % (elapsed))
# Parallel Thread XGBoost and CV
start = time.time()
model = XGBClassifier(nthread=-1)
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss', n_jobs=-1)
elapsed = time.time() - start
print("Parallel Thread XGBoost and CV: %f" % (elapsed))


# In[65]:


model2 = SVC()
model2.fit(X_train,y_train)


# In[62]:


print("the mean absolute error is",mean_absolute_error(y_test,model2.predict(X_test)))


# In[63]:


print("the accuracy of the train data is ",model2.score(X_train,y_train)*100)


# In[64]:


print("the accuracy of the test data is",model2.score(X_test,y_test)*100)


# In[ ]:





# In[ ]:




