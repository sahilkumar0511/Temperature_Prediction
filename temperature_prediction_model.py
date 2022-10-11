#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np


# In[37]:


import pandas as pd


# In[38]:


val = pd.read_csv('dataset.csv')


# In[39]:


s = val.dropna()


# In[40]:


x = s.drop('Temperature',axis=1)


# In[41]:


y = s['Temperature']


# In[42]:


get_ipython().system('pip install scikit-learn')


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


train_test_split(x, y, test_size=2)


# In[45]:


from sklearn.linear_model import LinearRegression


# In[46]:


model = LinearRegression()


# In[47]:


model.fit(x, y)


# In[48]:


model.score(x, y)


# In[49]:


ypred = model.predict(x)


# In[50]:


ypred


# In[51]:


import matplotlib.pyplot as plt


# In[53]:


s.plot(x = 'Temperature', y = 'Humidity',color = "yellow")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




