#!/usr/bin/env python
# coding: utf-8

# # Machine Learning

# ## Build Linear Regression Model using Diabetes Dataset (scikit learn)

# ### Import Libraries

# In[75]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


import pickle


# ### Load Dataset

# In[1]:


from sklearn.datasets import load_diabetes


# In[10]:


df = load_diabetes()
df


# In[13]:


dataset = pd.DataFrame(df.data)
dataset


# In[15]:


dataset.columns = df.feature_names


# In[16]:


dataset


# In[17]:


dataset.describe()


# In[18]:


df.target


# ## X and Y Data Matrices
# (Dependent and Independent features)

# In[21]:


x = dataset
y= df.target


# In[22]:


x.shape


# In[23]:


y.shape


# ## Dataset spliting: Trained and Test 
# (75/25 ratio)

# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state = 100)


# In[32]:


x_train


# ## Data Standardization

# In[36]:


scaler = StandardScaler()


# In[39]:


X_train = scaler.fit_transform(x_train)
X_train


# In[41]:


X_test = scaler.fit_transform(x_test)
X_test


# ## Data Dimensions

# In[76]:


X_train.shape, y_train.shape


# In[77]:


X_test.shape, y_test.shape


# ## Linear Regression Model

# ### Build a model

# In[43]:


model = LinearRegression()


# ### Train a model

# In[44]:


model.fit(X_train,y_train)


# ### Apply trained model to make predictions(using test dataset)

# In[45]:


y_pred = model.predict(X_test)
y_pred


# ## Prediction Results

# In[62]:


from sklearn.model_selection import cross_val_score


# In[67]:


mean_squared_score = cross_val_score(model, X_train,y_train, scoring='neg_mean_squared_error', cv= 5)
mean_squared_score


# In[68]:


np.mean(mean_squared_score)


# ### Model performance

# In[46]:


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


# ### Model Accuracy (R^2)

# In[69]:


model.score(X_test, y_test)


# In[61]:


from sklearn.metrics import r2_score

r2_score = r2_score(y_pred,y_test)
r2_score


# ### Adjusted R^2

# In[50]:


def adj_r2(x,y):
    r2 = model.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[51]:


adj_r2(X_test,y_test)


# ## Model Regularization Using ElasticNet

# In[52]:


elastic_cv= ElasticNetCV(alphas = None, cv = 10)
elastic_cv.fit(X_train, y_train)


# In[53]:


elastic_cv.alpha_


# In[54]:


elastic_cv.l1_ratio_


# In[55]:


elastic = ElasticNet(alpha =elastic_cv.alpha_,l1_ratio= elastic_cv.l1_ratio_)
elastic.fit(X_train, y_train)


# In[56]:


elastic.score(X_test,y_test)


# ### Save Model

# In[81]:


pickle.dump(model,open('Diabetes_LR_ML.pickle','wb'))  


# ## Data Visualization

# In[73]:


sns.displot(y_pred - y_test,kind ='kde')


# In[ ]:




