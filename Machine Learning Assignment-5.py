
# coding: utf-8

# In[1]:


# Problem Statement

#Build the random forest model after normalizing the variable to house pricing from boston data set.


# In[2]:


#Following the code to get data into the environment:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target


# In[3]:


features.head()


# In[4]:


targets


# In[5]:


# Visualizing Target Variables:

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(6, 4))
plt.hist(targets)
plt.xlabel('price ($100s)')
plt.ylabel('count')
plt.tight_layout()
plt.show()


# In[6]:


# Print the scatter plot for each feature with respect to price:

X = features.values
feature_names = features.columns
for index, feature_name in enumerate(features.columns):
    plt.figure(figsize=(4, 3))
    plt.scatter(X[:, index], targets)
    plt.ylabel('Price', size=15)
    plt.xlabel(feature_name, size=15)
    plt.tight_layout()


# In[7]:


# Split the data into trainning set and test set:

X = features.values
Y = targets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state=101)
print("X_train shape : ", X_train.shape)
print("X_test shape : ", X_test.shape)
print("Y_train shape : ", Y_train.shape)
print("Y_test shape : ", Y_test.shape)


# In[8]:


# Importance score of features used in RandomForest Regressor:

rf = RandomForestRegressor(random_state=1)
rf.fit(X, Y)
print("Features sorted by their score")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features.columns), reverse=True))
Y_pred = rf.predict(X_test)
print("Error Rate Of The Regression Model rf : ",mean_squared_error(Y_pred, Y_test))
print("R2 Score Of The Regression Model rf : ", r2_score(Y_pred, Y_test))


# In[9]:


lr = LinearRegression()
lr.fit(X, Y)
print("Features sorted their score:")

Y_pred_lr = lr.predict(X_test)
print("Error Rate of the Regression Model rf : ", mean_squared_error(Y_pred, Y_test))
print("R2 score of the Regression Model rf : ", r2_score(Y_pred_lr, Y_test))


# In[10]:


# Data Visualization:


# In[11]:


sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price($1000s)')
plt.tight_layout()
plt.title("Prices vs Predicted prices")


# In[12]:


# Data Visualization


# In[13]:


sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred_lr)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.title("Prices vs Predicted prices")


# In[14]:


# Using Bagging Mechanism to check the score of with different parameters.


# In[15]:


# we can achieve the above two tasks using the following codes
# Bagging: using all features


# In[16]:


rfc1 = RandomForestRegressor(max_features=13, random_state=1)
rfc1.fit(X_train, Y_train)
Y_pred1 = rfc1.predict(X_test)
print("Error Rate of the Regression Model rfc1 : ", mean_squared_error(Y_pred1, Y_test))
print("R2 Score of the Regression Model rfc1 : ", r2_score(Y_pred1, Y_test))
print('##########################################################################################')

# Play around with the setting for max_features:

rfc2 = RandomForestRegressor(max_features=8, random_state=1)
rfc2.fit(X_train, Y_train)
Y_pred2 = rfc2.predict(X_test)
print("Error Rate of the Regression Model rfc2 : ",mean_squared_error(Y_pred2,Y_test))
print("R2 Score of the Regression Model rfc2 : ",r2_score(Y_pred2,Y_test))
print('###########################################################################################')

# Play around with the setting for max_features:

rfc3 = RandomForestRegressor(n_estimators=20,max_features=8, random_state=1)
rfc3.fit(X_train, Y_train)
Y_pred3 = rfc2.predict(X_test)
print("Error Rate of the Regression Model rfc3 : ", mean_squared_error(Y_pred3, Y_test))
print("R2 Score of the Regression Model rfc3 : ", r2_score(Y_pred3, Y_test))


# In[17]:


print(sorted(zip(map(lambda x: round(x, 4), rfc1.feature_importances_), features.columns), reverse=True))


# In[18]:


print(sorted(zip(map(lambda x: round(x, 4), rfc2.feature_importances_), features.columns), reverse=True))


# In[19]:


print(sorted(zip(map(lambda x: round(x, 4), rfc3.feature_importances_), features.columns), reverse=True))


# In[20]:


df_corr = features.corr()
sns.set_style('whitegrid')
plt.figure(figsize=(20, 8))
sns.heatmap(df_corr, annot=True)


# In[21]:


#Create correlation matrix with absolute values:

df_corr = features.corr().abs()

# Select upper triangle of matrix

up_tri = df_corr.where(np.triu(np.ones(df_corr.shape[1]), k=1).astype(np.bool))


#Find all the features which have a correlation > 0.75 with other features.

corr_features = [ column for column in up_tri.columns if any(up_tri[column]> 0.75)]

# Print Correlated features:

print(corr_features)


# In[22]:


up_tri


# In[23]:


# Eliminating the correlated varaiables and trying the RandomForestRegressor again:


# In[24]:


# Eliminating two and keeping one features:

X1 = features.drop(['DIS', 'TAX'], axis=1)
Y1 = targets
X_train,X_test,Y_train,Y_test = train_test_split(X1,Y1,test_size =0.3,random_state=101)
print("X_train Shape : ",X_train.shape)
print("X_test Shape : ",X_test.shape)
print("Y_train Shape : ",Y_train.shape)
print("Y_test.shape : ",Y_test.shape)
rf4 = RandomForestRegressor()
rf4.fit(X_test, Y_test)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf4.feature_importances_), X1.columns),reverse=True))


# In[25]:


Y_pred4 = rf4.predict(X_test)
print("Error Rate of Regression Model rfc3 : ", mean_squared_error(Y_pred4, Y_test))
print("R2 Score of the Regression Model rfc3 : ", r2_score(Y_pred4, Y_test))


# In[26]:


# Regression Plot for RandomForestRegression Model rf4:

sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred4)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True Price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.title("Price vs Predicted prices ")


# In[27]:


# Regression Plot for RandomForestRegression Model rfc3:


# In[28]:


sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred3)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.title("Prices vs Predicted prices")


# In[29]:


# Regression Plot for RandomForestRegression Model rfc2:


# In[30]:


sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred2)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.title("Prices vs Predicted prices")


# In[31]:


# Regression Plot for RandomForestRegression Model rfc1:


# In[32]:


sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred1)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.title("Prices vs Predicted prices")


# In[33]:


# Regression Plot for RandomForestRegression Model rf:


# In[34]:


sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred)
plt.plot([0, 50], [0, 50], '--k')
plt.axis('tight')
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.tight_layout()
plt.title("Prices vs Predicted prices")


# In[35]:


# 10 Fold Cross Validation with max_features at 10:


# In[39]:


from sklearn.cross_validation import KFold
from collections import defaultdict
kfold = KFold(len(X),n_folds=10,shuffle=True,random_state=0)
lr = RandomForestRegressor(max_features=10,random_state=1)
X = features.values
Y = targets
names = features.columns
fold_accuracy = []
scores_kfold =  defaultdict(list)
for train_fold, valid_fold in kfold:
    train = X[train_fold] # Extract train data with cv indices
    valid = X[valid_fold] # Extract valid data with cv indices
    
    train_y = Y[train_fold]
    valid_y = Y[valid_fold]
    
    model = lr.fit(train,train_y)
    pred = rf.predict(valid)
    valid_acc = model.score(X = valid, y = valid_y)
    fold_accuracy.append(valid_acc) 
    acc = r2_score(valid_y, pred)
    for i in range(X.shape[1]):
        X_t = valid.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(valid_y, rf.predict(X_t))
        scores_kfold[names[i]].append((acc-shuff_acc)/acc)
    

print("Accuracy per fold: ", fold_accuracy, "\n")
print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))

print("Features sorted by their score:")
print(sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores_kfold.items()], reverse=True))


# In[40]:


# 10 FOLD CROSS VALDATON with max_features as 8:


# In[41]:


from sklearn.cross_validation import KFold
from collections import defaultdict
kfold = KFold(len(X),n_folds=8,shuffle=True,random_state=0)
lr = RandomForestRegressor(max_features=10,random_state=1)
X = features.values
Y = targets
names = features.columns
fold_accuracy = []
scores_kfold =  defaultdict(list)
for train_fold, valid_fold in kfold:
    train = X[train_fold] # Extract train data with cv indices
    valid = X[valid_fold] # Extract valid data with cv indices
    
    train_y = Y[train_fold]
    valid_y = Y[valid_fold]
    
    model = lr.fit(train,train_y)
    pred = rf.predict(valid)
    valid_acc = model.score(X = valid, y = valid_y)
    fold_accuracy.append(valid_acc) 
    acc = r2_score(valid_y, pred)
    for i in range(X.shape[1]):
        X_t = valid.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(valid_y, rf.predict(X_t))
        scores_kfold[names[i]].append((acc-shuff_acc)/acc)
    

print("Accuracy per fold: ", fold_accuracy, "\n")
print("Average accuracy: ", sum(fold_accuracy)/len(fold_accuracy))

print("Features sorted by their score:")
print(sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores_kfold.items()], reverse=True))

